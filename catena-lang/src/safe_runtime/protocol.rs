use std::io::{self, Read, Write};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use thiserror::Error;

use crate::{
    codegen::GpuDialect,
    runtime::{Backend, EntryPoint, ExecError},
};

pub(super) const MAX_FRAME_LEN: usize = 64 * 1024 * 1024;

pub(super) struct EncodedFrame {
    length: [u8; 4],
    payload: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) enum Request {
    Initialize {
        backend: Backend,
    },
    LoadSources {
        sources: Vec<String>,
    },
    AttachAsset {
        key: [u8; 32],
        byte_len: u64,
    },
    BindModel {
        binding: WireModelBinding,
    },
    StartGeneration {
        model: u64,
        capacity: u64,
    },
    StepGeneration {
        generation: u64,
        tokens: Vec<u32>,
    },
    ReleaseGeneration {
        generation: u64,
    },
    ReleaseModel {
        model: u64,
    },
    Execute {
        artifact: usize,
        name: String,
        buffers: Vec<WireIpcBuffer>,
        args: Vec<WireValue>,
    },
    ReleaseOutputs,
    Shutdown,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) enum Response {
    Initialized(Result<GpuDialect, String>),
    Loaded(Result<(usize, Vec<EntryPoint>), String>),
    Attached(Result<(u64, u64), String>),
    Resident(Result<ResidentResponse, String>),
    Executed(Result<WireExecution, RemoteExecError>),
    OutputsReleased,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) enum ResidentResponse {
    ModelBound(u64),
    GenerationStarted(u64),
    Token(u32),
    GenerationReleased(u64),
    ModelReleased(u64),
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct WireModelBinding {
    pub(super) artifact: usize,
    pub(super) entry_point: String,
    pub(super) assets: Vec<WireAssetSlice>,
    pub(super) state_byte_multipliers: Vec<u64>,
    pub(super) vocabulary_size: u64,
    pub(super) maximum_capacity: u64,
    pub(super) generation_device_allocation_budget_bytes: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(super) struct WireAssetSlice {
    pub(super) asset: u64,
    pub(super) offset: u64,
    pub(super) byte_len: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct WireExecution {
    pub(super) buffers: Vec<WireIpcBuffer>,
    pub(super) values: Vec<WireValue>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) enum RemoteExecError {
    Runtime(ExecError),
    Memory(String),
    Protocol(String),
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
    Encode(#[source] postcard::Error),
    #[error("failed to decode protocol message: {0}")]
    Decode(#[source] postcard::Error),
    #[error("protocol message has {remaining} trailing bytes")]
    TrailingBytes { remaining: usize },
    #[error("protocol frame is {actual} bytes, exceeding the {maximum}-byte limit")]
    FrameTooLarge { actual: usize, maximum: usize },
    #[error("protocol frame length {actual} cannot be represented on this platform")]
    FrameLengthUnsupported { actual: u32 },
}

#[derive(Debug, Error)]
pub(super) enum FrameEncodeError {
    #[error("failed to encode protocol message: {0}")]
    Encode(#[source] postcard::Error),
    #[error("protocol frame is {actual} bytes, exceeding the {maximum}-byte limit")]
    FrameTooLarge { actual: usize, maximum: usize },
}

impl From<FrameEncodeError> for ProtocolError {
    fn from(error: FrameEncodeError) -> Self {
        match error {
            FrameEncodeError::Encode(error) => Self::Encode(error),
            FrameEncodeError::FrameTooLarge { actual, maximum } => {
                Self::FrameTooLarge { actual, maximum }
            }
        }
    }
}

pub(super) fn encode_frame<T: Serialize>(message: &T) -> Result<EncodedFrame, FrameEncodeError> {
    let payload = postcard::to_allocvec(message).map_err(FrameEncodeError::Encode)?;
    if payload.len() > MAX_FRAME_LEN {
        return Err(FrameEncodeError::FrameTooLarge {
            actual: payload.len(),
            maximum: MAX_FRAME_LEN,
        });
    }
    let length = u32::try_from(payload.len()).map_err(|_| FrameEncodeError::FrameTooLarge {
        actual: payload.len(),
        maximum: MAX_FRAME_LEN,
    })?;
    Ok(EncodedFrame {
        length: length.to_le_bytes(),
        payload,
    })
}

pub(super) fn write_encoded_frame(
    writer: &mut impl Write,
    frame: &EncodedFrame,
) -> Result<(), ProtocolError> {
    writer.write_all(&frame.length)?;
    writer.write_all(&frame.payload)?;
    writer.flush()?;
    Ok(())
}

pub(super) fn write_frame<T: Serialize>(
    writer: &mut impl Write,
    message: &T,
) -> Result<(), ProtocolError> {
    let frame = encode_frame(message).map_err(ProtocolError::from)?;
    write_encoded_frame(writer, &frame)
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
    let wire_length = u32::from_le_bytes(length);
    let length =
        usize::try_from(wire_length).map_err(|_| ProtocolError::FrameLengthUnsupported {
            actual: wire_length,
        })?;
    if length > MAX_FRAME_LEN {
        return Err(ProtocolError::FrameTooLarge {
            actual: length,
            maximum: MAX_FRAME_LEN,
        });
    }
    let mut payload = vec![0_u8; length];
    reader.read_exact(&mut payload)?;
    let (message, remainder) =
        postcard::take_from_bytes(&payload).map_err(ProtocolError::Decode)?;
    if !remainder.is_empty() {
        return Err(ProtocolError::TrailingBytes {
            remaining: remainder.len(),
        });
    }
    Ok(Some(message))
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
    use crate::runtime::ValueKind;
    use serde::Serializer;

    struct RefusesToSerialize;

    impl Serialize for RefusesToSerialize {
        fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            Err(serde::ser::Error::custom("fixture rejected serialization"))
        }
    }

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
        let expected_entry = EntryPoint::new(
            "step".to_string(),
            vec![ValueKind::MemRef, ValueKind::MemOwn],
            vec![ValueKind::MemOwn],
        );
        let mut bytes = Vec::new();
        write_frame(
            &mut bytes,
            &Response::Loaded(Ok((expected, vec![expected_entry]))),
        )
        .unwrap();

        let Some(Response::Loaded(Ok((actual, entries)))) =
            read_frame(&mut bytes.as_slice()).unwrap()
        else {
            panic!("decoded the wrong response kind");
        };
        assert_eq!(actual, expected);
        assert_eq!(entries[0].name(), "step");
        assert_eq!(entries[0].inputs(), &[ValueKind::MemRef, ValueKind::MemOwn]);
        assert_eq!(entries[0].outputs(), &[ValueKind::MemOwn]);
    }

    #[test]
    fn asset_attachment_contains_only_fixed_identity_and_length() {
        let key = [0x5a; 32];
        let mut bytes = Vec::new();
        write_frame(
            &mut bytes,
            &Request::AttachAsset {
                key,
                byte_len: 4096,
            },
        )
        .unwrap();

        assert!(matches!(
            read_frame(&mut bytes.as_slice()).unwrap(),
            Some(Request::AttachAsset {
                key: actual_key,
                byte_len: 4096,
            }) if actual_key == key
        ));
    }

    #[test]
    fn resident_step_contains_only_generation_and_token_ids() {
        let mut bytes = Vec::new();
        write_frame(
            &mut bytes,
            &Request::StepGeneration {
                generation: 17,
                tokens: vec![3, u32::MAX],
            },
        )
        .unwrap();

        assert!(matches!(
            read_frame(&mut bytes.as_slice()).unwrap(),
            Some(Request::StepGeneration {
                generation: 17,
                tokens,
            }) if tokens == [3, u32::MAX]
        ));
    }

    #[test]
    fn model_binding_preserves_the_provider_allocation_envelope() {
        let mut bytes = Vec::new();
        write_frame(
            &mut bytes,
            &Request::BindModel {
                binding: WireModelBinding {
                    artifact: artifact(),
                    entry_point: "forward".into(),
                    assets: Vec::new(),
                    state_byte_multipliers: vec![4, 8],
                    vocabulary_size: 32,
                    maximum_capacity: 64,
                    generation_device_allocation_budget_bytes: 123_456,
                },
            },
        )
        .unwrap();

        let Some(Request::BindModel { binding }) = read_frame(&mut bytes.as_slice()).unwrap()
        else {
            panic!("decoded the wrong request kind");
        };
        assert_eq!(binding.generation_device_allocation_budget_bytes, 123_456);
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
    fn structured_execution_rejections_round_trip() {
        let mut bytes = Vec::new();
        write_frame(
            &mut bytes,
            &Response::Executed(Err(RemoteExecError::Memory("invalid view".into()))),
        )
        .unwrap();
        write_frame(
            &mut bytes,
            &Response::Executed(Err(RemoteExecError::Protocol("pending outputs".into()))),
        )
        .unwrap();

        let mut bytes = bytes.as_slice();
        assert!(matches!(
            read_frame(&mut bytes).unwrap(),
            Some(Response::Executed(Err(RemoteExecError::Memory(error))))
                if error == "invalid view"
        ));
        assert!(matches!(
            read_frame(&mut bytes).unwrap(),
            Some(Response::Executed(Err(RemoteExecError::Protocol(error))))
                if error == "pending outputs"
        ));
    }

    #[test]
    fn teardown_acknowledgements_round_trip_with_the_released_identity() {
        let mut bytes = Vec::new();
        write_frame(
            &mut bytes,
            &Response::Resident(Ok(ResidentResponse::GenerationReleased(17))),
        )
        .unwrap();
        write_frame(
            &mut bytes,
            &Response::Resident(Ok(ResidentResponse::ModelReleased(23))),
        )
        .unwrap();
        write_frame(&mut bytes, &Response::OutputsReleased).unwrap();

        let mut bytes = bytes.as_slice();
        assert!(matches!(
            read_frame(&mut bytes).unwrap(),
            Some(Response::Resident(Ok(
                ResidentResponse::GenerationReleased(17)
            )))
        ));
        assert!(matches!(
            read_frame(&mut bytes).unwrap(),
            Some(Response::Resident(Ok(ResidentResponse::ModelReleased(23))))
        ));
        assert!(matches!(
            read_frame(&mut bytes).unwrap(),
            Some(Response::OutputsReleased)
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
    fn trailing_payload_bytes_are_an_error() {
        let mut payload = postcard::to_allocvec(&Request::Shutdown).unwrap();
        payload.push(0);
        let mut bytes = Vec::from(u32::try_from(payload.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(&payload);

        let error = read_frame::<Request>(&mut bytes.as_slice()).unwrap_err();
        assert!(matches!(
            error,
            ProtocolError::TrailingBytes { remaining: 1 }
        ));
    }

    #[test]
    fn encoding_failure_writes_no_frame_bytes() {
        let mut bytes = Vec::new();
        let error = write_frame(&mut bytes, &RefusesToSerialize).unwrap_err();
        assert!(matches!(error, ProtocolError::Encode(_)));
        assert!(bytes.is_empty());
    }

    #[test]
    fn rejects_oversized_frame_before_allocating() {
        let length = u32::try_from(MAX_FRAME_LEN + 1).unwrap().to_le_bytes();
        let error = read_frame::<Request>(&mut length.as_slice()).unwrap_err();
        assert!(matches!(error, ProtocolError::FrameTooLarge { .. }));
    }
}
