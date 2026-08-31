use std::io::{self, Read, Write};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use thiserror::Error;

use crate::{codegen::GpuDialect, runtime::ExecError};

const MAX_FRAME_LEN: usize = 64 * 1024 * 1024;

#[derive(Debug, Serialize, Deserialize)]
pub(super) enum Request {
    Initialize {
        dialect: GpuDialect,
    },
    LoadSources {
        sources: Vec<String>,
    },
    Execute {
        artifact: usize,
        name: String,
        buffers: Vec<WireIpcBuffer>,
        args: Vec<WireValue>,
    },
    ReleaseArtifact {
        artifact: usize,
    },
    ReleaseOutputs,
    Shutdown,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) enum Response {
    Initialized(Result<(), String>),
    Loaded(Result<usize, String>),
    Executed(Result<WireExecution, RemoteExecError>),
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct WireExecution {
    pub(super) buffers: Vec<WireIpcBuffer>,
    pub(super) values: Vec<WireValue>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) enum RemoteExecError {
    Runtime(ExecError),
    UnknownArtifact,
    Memory(String),
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) enum WireValue {
    Bool(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(f32),
    MemRef {
        buffer: usize,
        view_offset: u64,
        byte_len: u64,
    },
    MemOwn {
        buffer: usize,
        view_offset: u64,
        byte_len: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct WireIpcBuffer {
    pub(super) handle: Option<Vec<u8>>,
    pub(super) allocation_byte_len: u64,
}

#[derive(Debug, Error)]
pub(super) enum ProtocolError {
    #[error("protocol I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("failed to encode protocol message: {0}")]
    Encode(#[source] Box<bincode::ErrorKind>),
    #[error("failed to decode protocol message: {0}")]
    Decode(#[source] Box<bincode::ErrorKind>),
    #[error("protocol frame is {actual} bytes, exceeding the {maximum}-byte limit")]
    FrameTooLarge { actual: usize, maximum: usize },
}

pub(super) fn write_frame<T: Serialize>(
    writer: &mut impl Write,
    message: &T,
) -> Result<(), ProtocolError> {
    let payload = bincode::serialize(message).map_err(ProtocolError::Encode)?;
    if payload.len() > MAX_FRAME_LEN || payload.len() > u32::MAX as usize {
        return Err(ProtocolError::FrameTooLarge {
            actual: payload.len(),
            maximum: MAX_FRAME_LEN,
        });
    }
    writer.write_all(&(payload.len() as u32).to_le_bytes())?;
    writer.write_all(&payload)?;
    writer.flush()?;
    Ok(())
}

pub(super) fn read_frame<T: DeserializeOwned>(
    reader: &mut impl Read,
) -> Result<Option<T>, ProtocolError> {
    let Some(first) = read_first_byte(reader)? else {
        return Ok(None);
    };
    let mut length = [0_u8; 4];
    length[0] = first;
    reader.read_exact(&mut length[1..])?;
    let length = u32::from_le_bytes(length) as usize;
    if length > MAX_FRAME_LEN {
        return Err(ProtocolError::FrameTooLarge {
            actual: length,
            maximum: MAX_FRAME_LEN,
        });
    }
    let mut payload = vec![0_u8; length];
    reader.read_exact(&mut payload)?;
    bincode::deserialize(&payload)
        .map(Some)
        .map_err(ProtocolError::Decode)
}

fn read_first_byte(reader: &mut impl Read) -> Result<Option<u8>, io::Error> {
    let mut byte = [0_u8; 1];
    loop {
        match reader.read(&mut byte) {
            Ok(0) => return Ok(None),
            Ok(_) => return Ok(Some(byte[0])),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact() -> usize {
        7
    }

    #[test]
    fn frames_round_trip() {
        let expected_artifact = artifact();
        let mut bytes = Vec::new();
        write_frame(
            &mut bytes,
            &Request::Execute {
                artifact: expected_artifact,
                name: "f".to_string(),
                buffers: Vec::new(),
                args: vec![WireValue::U64(7)],
            },
        )
        .unwrap();

        let decoded: Request = read_frame(&mut bytes.as_slice()).unwrap().unwrap();
        assert!(matches!(
            decoded,
            Request::Execute {
                artifact,
                name,
                buffers,
                args,
            } if artifact == expected_artifact && name == "f" && buffers.is_empty()
                && matches!(args.as_slice(), [WireValue::U64(7)])
        ));
    }

    #[test]
    fn loaded_artifact_round_trips() {
        let expected = artifact();
        let mut bytes = Vec::new();
        write_frame(&mut bytes, &Response::Loaded(Ok(expected))).unwrap();

        assert!(matches!(
            read_frame(&mut bytes.as_slice()).unwrap(),
            Some(Response::Loaded(Ok(actual))) if actual == expected
        ));
    }

    #[test]
    fn artifact_release_round_trips() {
        let expected = artifact();
        let mut bytes = Vec::new();
        write_frame(&mut bytes, &Request::ReleaseArtifact { artifact: expected }).unwrap();

        assert!(matches!(
            read_frame(&mut bytes.as_slice()).unwrap(),
            Some(Request::ReleaseArtifact { artifact }) if artifact == expected
        ));
    }

    #[test]
    fn memory_request_uses_a_buffer_table_index() {
        let mut bytes = Vec::new();
        write_frame(
            &mut bytes,
            &Request::Execute {
                artifact: artifact(),
                name: "head".to_string(),
                buffers: vec![WireIpcBuffer {
                    handle: Some(vec![7; 64]),
                    allocation_byte_len: 1024,
                }],
                args: vec![WireValue::MemRef {
                    buffer: 0,
                    view_offset: 16,
                    byte_len: 32,
                }],
            },
        )
        .unwrap();

        let Request::Execute { buffers, args, .. } = read_frame::<Request>(&mut bytes.as_slice())
            .unwrap()
            .unwrap()
        else {
            panic!("decoded the wrong request kind");
        };
        assert_eq!(buffers[0].handle.as_deref(), Some(&[7; 64][..]));
        assert!(matches!(
            args.as_slice(),
            [WireValue::MemRef {
                buffer: 0,
                view_offset: 16,
                byte_len: 32,
            }]
        ));
    }

    #[test]
    fn owned_output_and_release_round_trip() {
        let execution = Response::Executed(Ok(WireExecution {
            buffers: vec![WireIpcBuffer {
                handle: Some(vec![9; 64]),
                allocation_byte_len: 256,
            }],
            values: vec![WireValue::MemOwn {
                buffer: 0,
                view_offset: 0,
                byte_len: 256,
            }],
        }));
        let mut bytes = Vec::new();
        write_frame(&mut bytes, &execution).unwrap();
        write_frame(&mut bytes, &Request::ReleaseOutputs).unwrap();

        let mut bytes = bytes.as_slice();
        let Some(Response::Executed(Ok(execution))) = read_frame(&mut bytes).unwrap() else {
            panic!("decoded the wrong response kind");
        };
        assert!(matches!(
            execution.values.as_slice(),
            [WireValue::MemOwn {
                buffer: 0,
                view_offset: 0,
                byte_len: 256,
            }]
        ));
        assert!(matches!(
            read_frame(&mut bytes).unwrap(),
            Some(Request::ReleaseOutputs)
        ));
    }

    #[test]
    fn clean_eof_has_no_frame() {
        let result = read_frame::<Request>(&mut &[][..]).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn partial_header_is_an_error() {
        let error = read_frame::<Request>(&mut &[1, 0][..]).unwrap_err();
        assert!(matches!(error, ProtocolError::Io(_)));
    }

    #[test]
    fn malformed_payload_is_an_error() {
        let mut bytes = Vec::from(1_u32.to_le_bytes());
        bytes.push(0xff);
        let error = read_frame::<Request>(&mut bytes.as_slice()).unwrap_err();
        assert!(matches!(error, ProtocolError::Decode(_)));
    }

    #[test]
    fn rejects_oversized_frame_before_allocating() {
        let length = ((MAX_FRAME_LEN + 1) as u32).to_le_bytes();
        let error = read_frame::<Request>(&mut length.as_slice()).unwrap_err();
        assert!(matches!(error, ProtocolError::FrameTooLarge { .. }));
    }
}
