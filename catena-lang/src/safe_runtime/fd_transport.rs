use std::{
    fs::File,
    mem::MaybeUninit,
    os::fd::{AsFd, BorrowedFd, OwnedFd},
    os::unix::net::UnixStream,
};

use rustix::{
    io::{IoSlice, IoSliceMut, retry_on_intr},
    net::{
        RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags, ReturnFlags, SendAncillaryBuffer,
        SendAncillaryMessage, SendFlags, recvmsg, sendmsg,
    },
};
use thiserror::Error;

const DESCRIPTOR_MARKER: u8 = 0xa7;
const MAX_DESCRIPTORS_PER_MESSAGE: usize = 2;

#[derive(Debug, Error)]
pub(super) enum FdTransportError {
    #[error("failed to send asset descriptor: {0}")]
    Send(#[source] rustix::io::Errno),
    #[error("failed to receive asset descriptor: {0}")]
    Receive(#[source] rustix::io::Errno),
    #[error("asset descriptor message was truncated")]
    Truncated,
    #[error("asset descriptor message has an invalid payload")]
    InvalidPayload,
    #[error("asset descriptor message contains unexpected ancillary data")]
    UnexpectedAncillary,
    #[error("asset descriptor message contains {actual} descriptors, expected exactly one")]
    DescriptorCount { actual: usize },
    #[error("asset descriptor control buffer is too small")]
    ControlBuffer,
}

pub(super) fn send_file(socket: &UnixStream, file: &File) -> Result<(), FdTransportError> {
    send_fds(socket, &[file.as_fd()])
}

fn send_fds(socket: &UnixStream, descriptors: &[BorrowedFd<'_>]) -> Result<(), FdTransportError> {
    let payload = [DESCRIPTOR_MARKER];
    let iov = [IoSlice::new(&payload)];
    let mut space =
        [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(MAX_DESCRIPTORS_PER_MESSAGE))];
    let mut ancillary = SendAncillaryBuffer::new(&mut space);
    if !ancillary.push(SendAncillaryMessage::ScmRights(descriptors)) {
        return Err(FdTransportError::ControlBuffer);
    }
    let sent = retry_on_intr(|| sendmsg(socket, &iov, &mut ancillary, SendFlags::NOSIGNAL))
        .map_err(FdTransportError::Send)?;
    if sent != payload.len() {
        return Err(FdTransportError::InvalidPayload);
    }
    Ok(())
}

pub(super) fn receive_file(socket: &UnixStream) -> Result<File, FdTransportError> {
    let mut payload = [0_u8];
    let mut iov = [IoSliceMut::new(&mut payload)];
    let mut space =
        [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(MAX_DESCRIPTORS_PER_MESSAGE))];
    let mut ancillary = RecvAncillaryBuffer::new(&mut space);
    let received =
        retry_on_intr(|| recvmsg(socket, &mut iov, &mut ancillary, RecvFlags::CMSG_CLOEXEC))
            .map_err(FdTransportError::Receive)?;
    if received
        .flags
        .intersects(ReturnFlags::TRUNC | ReturnFlags::CTRUNC)
    {
        return Err(FdTransportError::Truncated);
    }
    if received.bytes != payload.len() || payload[0] != DESCRIPTOR_MARKER {
        return Err(FdTransportError::InvalidPayload);
    }

    let mut descriptors = Vec::<OwnedFd>::new();
    let mut unexpected = false;
    for message in ancillary.drain() {
        match message {
            RecvAncillaryMessage::ScmRights(rights) => descriptors.extend(rights),
            _ => unexpected = true,
        }
    }
    if unexpected {
        return Err(FdTransportError::UnexpectedAncillary);
    }
    if descriptors.len() != 1 {
        return Err(FdTransportError::DescriptorCount {
            actual: descriptors.len(),
        });
    }
    Ok(File::from(
        descriptors.pop().expect("descriptor count was checked"),
    ))
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Seek, Write};
    use std::os::fd::AsRawFd;

    use super::*;

    #[test]
    fn transfers_exactly_one_cloexec_descriptor() {
        let (sender, receiver) = UnixStream::pair().unwrap();
        let mut source = tempfile::tempfile().unwrap();
        source.write_all(b"asset bytes").unwrap();
        source.rewind().unwrap();

        send_file(&sender, &source).unwrap();
        let mut received = receive_file(&receiver).unwrap();
        let flags = unsafe { libc::fcntl(received.as_fd().as_raw_fd(), libc::F_GETFD) };
        assert_ne!(flags, -1);
        assert_ne!(flags & libc::FD_CLOEXEC, 0);
        let mut bytes = Vec::new();
        received.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes, b"asset bytes");
    }

    #[test]
    fn rejects_missing_descriptor() {
        let (mut sender, receiver) = UnixStream::pair().unwrap();
        sender.write_all(&[DESCRIPTOR_MARKER]).unwrap();
        assert!(matches!(
            receive_file(&receiver),
            Err(FdTransportError::DescriptorCount { actual: 0 })
        ));
    }

    #[test]
    fn rejects_wrong_payload_even_with_a_descriptor() {
        let (sender, receiver) = UnixStream::pair().unwrap();
        let source = tempfile::tempfile().unwrap();
        let payload = [DESCRIPTOR_MARKER.wrapping_add(1)];
        let iov = [IoSlice::new(&payload)];
        let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(1))];
        let mut ancillary = SendAncillaryBuffer::new(&mut space);
        let descriptors = [source.as_fd()];
        assert!(ancillary.push(SendAncillaryMessage::ScmRights(&descriptors)));
        sendmsg(&sender, &iov, &mut ancillary, SendFlags::NOSIGNAL).unwrap();
        assert!(matches!(
            receive_file(&receiver),
            Err(FdTransportError::InvalidPayload)
        ));
    }

    #[test]
    fn rejects_multiple_descriptors() {
        let (sender, receiver) = UnixStream::pair().unwrap();
        let first = tempfile::tempfile().unwrap();
        let second = tempfile::tempfile().unwrap();
        send_fds(&sender, &[first.as_fd(), second.as_fd()]).unwrap();
        assert!(matches!(
            receive_file(&receiver),
            Err(FdTransportError::DescriptorCount { actual: 2 })
        ));
    }
}
