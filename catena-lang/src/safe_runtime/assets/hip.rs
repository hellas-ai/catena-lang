use std::{
    env,
    ffi::{CStr, c_char, c_int, c_uint, c_void},
    path::PathBuf,
    ptr::NonNull,
};

use anyhow::{Context, Result, bail};
use libloading::{Library, Symbol};
use memmap2::Mmap;

const HIP_HOST_REGISTER_MAPPED: c_uint = 0x2;
const HIP_EXT_HOST_REGISTER_COARSE_GRAINED: c_uint = 0x8;

type HipHostRegister = unsafe extern "C" fn(*mut c_void, usize, c_uint) -> c_int;
type HipHostGetDevicePointer = unsafe extern "C" fn(*mut *mut c_void, *mut c_void, c_uint) -> c_int;
type HipHostUnregister = unsafe extern "C" fn(*mut c_void) -> c_int;
type HipGetErrorString = unsafe extern "C" fn(c_int) -> *const c_char;

pub(super) struct HipRegistration {
    host_base: NonNull<c_void>,
    pub(super) device_base: NonNull<c_void>,
    unregister: HipHostUnregister,
    error_string: HipGetErrorString,
    _library: Library,
}

impl HipRegistration {
    pub(super) fn new(bytes: &Mmap) -> Result<Self> {
        let library = load_hip_library()?;
        let register: HipHostRegister = load_symbol(&library, b"hipHostRegister\0")?;
        let get_device_pointer: HipHostGetDevicePointer =
            load_symbol(&library, b"hipHostGetDevicePointer\0")?;
        let unregister: HipHostUnregister = load_symbol(&library, b"hipHostUnregister\0")?;
        let error_string: HipGetErrorString = load_symbol(&library, b"hipGetErrorString\0")?;
        let host_base = NonNull::new(bytes.as_ptr().cast_mut().cast::<c_void>())
            .context("mmap returned a null pointer")?;
        let flags = HIP_HOST_REGISTER_MAPPED | HIP_EXT_HOST_REGISTER_COARSE_GRAINED;
        let status = unsafe { register(host_base.as_ptr(), bytes.len(), flags) };
        if status != 0 {
            bail!(
                "hipHostRegister failed: {}",
                hip_error(error_string, status)
            );
        }
        let mut device_base = std::ptr::null_mut();
        let status = unsafe { get_device_pointer(&mut device_base, host_base.as_ptr(), 0) };
        if status != 0 {
            unsafe { unregister(host_base.as_ptr()) };
            bail!(
                "hipHostGetDevicePointer failed: {}",
                hip_error(error_string, status)
            );
        }
        let Some(device_base) = NonNull::new(device_base) else {
            unsafe { unregister(host_base.as_ptr()) };
            bail!("hipHostGetDevicePointer returned a null pointer");
        };
        Ok(Self {
            host_base,
            device_base,
            unregister,
            error_string,
            _library: library,
        })
    }
}

impl Drop for HipRegistration {
    fn drop(&mut self) {
        let status = unsafe { (self.unregister)(self.host_base.as_ptr()) };
        if status != 0 {
            eprintln!(
                "warning: hipHostUnregister failed while dropping mapped asset: {}",
                hip_error(self.error_string, status)
            );
        }
    }
}

fn load_hip_library() -> Result<Library> {
    let mut paths = vec![PathBuf::from("libamdhip64.so")];
    for variable in ["ROCM_PATH", "HIP_PATH"] {
        if let Some(root) = env::var_os(variable) {
            let path = PathBuf::from(root).join("lib/libamdhip64.so");
            if !paths.contains(&path) {
                paths.push(path);
            }
        }
    }
    let mut errors = Vec::new();
    for path in &paths {
        match unsafe { Library::new(path) } {
            Ok(library) => return Ok(library),
            Err(error) => errors.push(format!("{}: {error}", path.display())),
        }
    }
    bail!("failed to load HIP runtime library ({})", errors.join(", "))
}

fn load_symbol<T: Copy>(library: &Library, symbol: &'static [u8]) -> Result<T> {
    let loaded: Symbol<'_, T> = unsafe { library.get(symbol) }.with_context(|| {
        format!(
            "failed to resolve HIP symbol {}",
            String::from_utf8_lossy(symbol).trim_end_matches('\0')
        )
    })?;
    Ok(*loaded)
}

fn hip_error(error_string: HipGetErrorString, status: c_int) -> String {
    let pointer = unsafe { error_string(status) };
    if pointer.is_null() {
        format!("status {status}")
    } else {
        format!(
            "{} (status {status})",
            unsafe { CStr::from_ptr(pointer) }.to_string_lossy()
        )
    }
}
