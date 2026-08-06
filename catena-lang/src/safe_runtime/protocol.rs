use std::io::{self, Read, Write};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use thiserror::Error;

use crate::{codegen::GpuDialect, runtime::ExecError};

const MAX_FRAME_LEN: usize = 64 * 1024 * 1024;

#[derive(Debug, Serialize, Deserialize)]
pub(crate) enum Request {
    Initialize {
        sources: Vec<String>,
        dialect: WireGpuDialect,
    },
    Execute {
        execution_id: u64,
        name: String,
        args: Vec<WireValue>,
    },
    Acknowledge {
        execution_id: u64,
    },
    Shutdown,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) enum Response {
    Initialized(Result<(), String>),
    Executed {
        execution_id: u64,
        result: Result<Vec<WireValue>, RemoteExecError>,
    },
    Acknowledged {
        execution_id: u64,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) enum RemoteExecError {
    Runtime(ExecError),
    Memory(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum WireGpuDialect {
    Hip,
    Cuda,
}

impl From<GpuDialect> for WireGpuDialect {
    fn from(value: GpuDialect) -> Self {
        match value {
            GpuDialect::Hip => Self::Hip,
            GpuDialect::Cuda => Self::Cuda,
        }
    }
}

impl From<WireGpuDialect> for GpuDialect {
    fn from(value: WireGpuDialect) -> Self {
        match value {
            WireGpuDialect::Hip => Self::Hip,
            WireGpuDialect::Cuda => Self::Cuda,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum WireCapability {
    Own,
    Ref,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct WireMemory {
    pub(crate) capability: WireCapability,
    pub(crate) dialect: WireGpuDialect,
    pub(crate) byte_len: u64,
    pub(crate) handle: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) enum WireValue {
    Bool(u8),
    U32(u32),
    U64(u64),
    F32(f32),
    Mem(WireMemory),
}

#[derive(Debug, Error)]
pub(crate) enum ProtocolError {
    #[error("protocol I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("failed to encode protocol message: {0}")]
    Encode(#[source] Box<bincode::ErrorKind>),
    #[error("failed to decode protocol message: {0}")]
    Decode(#[source] Box<bincode::ErrorKind>),
    #[error("protocol frame is {actual} bytes, exceeding the {maximum}-byte limit")]
    FrameTooLarge { actual: usize, maximum: usize },
}

pub(crate) fn write_frame<T: Serialize>(
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

pub(crate) fn read_frame<T: DeserializeOwned>(
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

    #[test]
    fn memory_execute_and_acknowledgement_round_trip() {
        let mut bytes = Vec::new();
        write_frame(
            &mut bytes,
            &Request::Execute {
                execution_id: 7,
                name: "identity".to_string(),
                args: vec![WireValue::Mem(WireMemory {
                    capability: WireCapability::Own,
                    dialect: WireGpuDialect::Hip,
                    byte_len: 16,
                    handle: vec![3; 64],
                })],
            },
        )
        .unwrap();
        write_frame(&mut bytes, &Request::Acknowledge { execution_id: 7 }).unwrap();

        let mut bytes = bytes.as_slice();
        let Request::Execute {
            execution_id, args, ..
        } = read_frame(&mut bytes).unwrap().unwrap()
        else {
            panic!("decoded the wrong request kind");
        };
        assert_eq!(execution_id, 7);
        assert!(matches!(
            args.as_slice(),
            [WireValue::Mem(WireMemory {
                capability: WireCapability::Own,
                byte_len: 16,
                ..
            })]
        ));
        assert!(matches!(
            read_frame(&mut bytes).unwrap(),
            Some(Request::Acknowledge { execution_id: 7 })
        ));
    }

    #[test]
    fn clean_eof_has_no_frame() {
        assert!(read_frame::<Request>(&mut &[][..]).unwrap().is_none());
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
