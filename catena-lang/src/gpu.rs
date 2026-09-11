use std::{
    env,
    ffi::{CString, c_int, c_uint, c_void},
    path::PathBuf,
    ptr::NonNull,
    sync::{Arc, OnceLock},
};

use libloading::{Library, Symbol};

use crate::codegen::GpuDialect;
use crate::runtime::MemError;

const MEMCPY_HOST_TO_DEVICE: c_int = 1;
const MEMCPY_DEVICE_TO_HOST: c_int = 2;
const MEMCPY_DEVICE_TO_DEVICE: c_int = 3;
const IPC_LAZY_ENABLE_PEER_ACCESS: c_uint = 1;
const DEVICE_MAP_HOST: c_uint = 0x8;
const HOST_REGISTER_MAPPED: c_uint = 0x2;
const HIP_HOST_REGISTER_COARSE_GRAINED: c_uint = 0x8;
const CUDA_HOST_REGISTER_READ_ONLY: c_uint = 0x8;
pub(crate) const IPC_HANDLE_BYTES: usize = 64;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RawIpcMemHandle {
    pub(crate) bytes: [u8; IPC_HANDLE_BYTES],
}

/// The process-local HIP/CUDA operations needed to manage a [`crate::runtime::MemOwn`].
///
/// Transport-specific operations, including IPC and device-to-device copies,
/// deliberately live outside Runtime.
#[derive(Debug)]
pub(crate) struct GpuApi {
    dialect: GpuDialect,
    runtime_library: Library,
    cuda_driver_library: Option<Library>,
}

impl GpuApi {
    pub(crate) fn load(dialect: GpuDialect) -> Result<Arc<Self>, MemError> {
        static HIP: OnceLock<Arc<GpuApi>> = OnceLock::new();
        static CUDA: OnceLock<Arc<GpuApi>> = OnceLock::new();

        let cached = match dialect {
            GpuDialect::Hip => &HIP,
            GpuDialect::Cuda => &CUDA,
        };
        if let Some(gpu) = cached.get() {
            return Ok(gpu.clone());
        }

        let gpu = Arc::new(Self::load_uncached(dialect)?);
        Ok(cached.get_or_init(|| gpu).clone())
    }

    fn load_uncached(dialect: GpuDialect) -> Result<Self, MemError> {
        let library_name = match dialect {
            GpuDialect::Hip => "libamdhip64.so",
            GpuDialect::Cuda => "libcudart.so",
        };
        let mut tried = vec![PathBuf::from(library_name)];
        tried.extend(runtime_library_fallback_paths(dialect));
        let runtime_library = load_library(dialect, tried)?;
        let cuda_driver_library = match dialect {
            GpuDialect::Hip => None,
            GpuDialect::Cuda => Some(load_library(
                dialect,
                vec![PathBuf::from("libcuda.so.1"), PathBuf::from("libcuda.so")],
            )?),
        };
        let api = Self {
            dialect,
            runtime_library,
            cuda_driver_library,
        };
        api.initialize_device()?;
        Ok(api)
    }

    fn initialize_device(&self) -> Result<(), MemError> {
        let count: Symbol<'_, unsafe extern "C" fn(*mut c_int) -> c_int> =
            unsafe { self.load_symbol(self.symbol("hipGetDeviceCount", "cudaGetDeviceCount"))? };
        let mut devices = 0;
        self.check("enumerate devices", unsafe { count(&mut devices) })?;
        if devices == 0 {
            return Err(MemError::NoDevice {
                dialect: self.dialect,
            });
        }
        // Configure mapped host memory before CUDA's primary context is created.
        // Vendor visibility variables determine which physical GPU is device 0.
        let flags: Symbol<'_, unsafe extern "C" fn(c_uint) -> c_int> =
            unsafe { self.load_symbol(self.symbol("hipSetDeviceFlags", "cudaSetDeviceFlags"))? };
        self.check("enable mapped host memory", unsafe {
            flags(DEVICE_MAP_HOST)
        })?;
        let select: Symbol<'_, unsafe extern "C" fn(c_int) -> c_int> =
            unsafe { self.load_symbol(self.symbol("hipSetDevice", "cudaSetDevice"))? };
        self.check("select device", unsafe { select(0) })
    }

    pub(crate) fn supports_read_only_host(&self) -> Result<bool, MemError> {
        if self.dialect == GpuDialect::Hip {
            return Ok(true);
        }
        let attribute: Symbol<'_, unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int> =
            unsafe { self.load_symbol("cudaDeviceGetAttribute")? };
        let mut supported = 0;
        self.check("query read-only host registration", unsafe {
            attribute(&mut supported, 113, 0)
        })?;
        Ok(supported != 0)
    }

    /// Register a host mapping without a device allocation.
    ///
    /// # Safety
    /// The range must remain mapped until unregister_host succeeds or the
    /// process exits, and must not already be registered with this runtime.
    /// When read_only is false, the mapping must be privately writable.
    pub(crate) unsafe fn register_host(
        &self,
        base: *mut c_void,
        byte_len: usize,
        read_only: bool,
    ) -> Result<NonNull<c_void>, MemError> {
        let register: Symbol<'_, unsafe extern "C" fn(*mut c_void, usize, c_uint) -> c_int> =
            unsafe { self.load_symbol(self.symbol("hipHostRegister", "cudaHostRegister"))? };
        let pointer: Symbol<
            '_,
            unsafe extern "C" fn(*mut *mut c_void, *mut c_void, c_uint) -> c_int,
        > = unsafe {
            self.load_symbol(self.symbol("hipHostGetDevicePointer", "cudaHostGetDevicePointer"))?
        };
        // Resolve cleanup before registering, so later symbol errors cannot
        // strand a pinned mapping. Bit 0x8 means coarse-grained on HIP and
        // read-only on CUDA; CUDA requires it for a read-only CPU mmap.
        let unregister: Symbol<'_, unsafe extern "C" fn(*mut c_void) -> c_int> =
            unsafe { self.load_symbol(self.symbol("hipHostUnregister", "cudaHostUnregister"))? };
        let flags = HOST_REGISTER_MAPPED
            | match self.dialect {
                GpuDialect::Hip => HIP_HOST_REGISTER_COARSE_GRAINED,
                GpuDialect::Cuda if read_only => CUDA_HOST_REGISTER_READ_ONLY,
                GpuDialect::Cuda => 0,
            };
        self.check("register read-only host memory", unsafe {
            register(base, byte_len, flags)
        })?;
        let mut device = std::ptr::null_mut();
        let result = self
            .check("map registered host memory", unsafe {
                pointer(&mut device, base, 0)
            })
            .and_then(|()| {
                NonNull::new(device).ok_or(MemError::NullMappedPointer {
                    dialect: self.dialect,
                })
            });
        if result.is_err()
            && let Err(error) = self.check("unregister host memory after mapping failure", unsafe {
                unregister(base)
            })
        {
            // The caller owns the mmap. It cannot safely drop it while
            // a failed rollback leaves the GPU registration live.
            eprintln!("fatal GPU mapping cleanup failure: {error}");
            std::process::abort();
        }
        result
    }

    /// # Safety
    /// base must be a live mapping registered by this GPU API.
    pub(crate) unsafe fn unregister_host(&self, base: *mut c_void) -> Result<(), MemError> {
        let function: Symbol<'_, unsafe extern "C" fn(*mut c_void) -> c_int> =
            unsafe { self.load_symbol(self.symbol("hipHostUnregister", "cudaHostUnregister"))? };
        self.check("unregister host memory", unsafe { function(base) })
    }

    pub(crate) fn dialect(&self) -> GpuDialect {
        self.dialect
    }

    pub(crate) fn allocate(&self, byte_len: usize) -> Result<*mut c_void, MemError> {
        if byte_len == 0 {
            return Ok(std::ptr::null_mut());
        }

        let mut data = std::ptr::null_mut();
        let symbol = self.symbol("hipMalloc", "cudaMalloc");
        let function: Symbol<'_, unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int> =
            unsafe { self.load_symbol(symbol)? };
        self.check("allocate device memory", unsafe {
            function(&mut data, byte_len)
        })?;
        Ok(data)
    }

    pub(crate) fn copy_host_to_device(
        &self,
        destination: *mut c_void,
        source: &[u8],
    ) -> Result<(), MemError> {
        if source.is_empty() {
            return Ok(());
        }
        self.copy(
            destination,
            source.as_ptr().cast(),
            source.len(),
            MEMCPY_HOST_TO_DEVICE,
            "copy host to device",
        )
    }

    pub(crate) fn zero(&self, destination: *mut c_void, byte_len: usize) -> Result<(), MemError> {
        if byte_len == 0 {
            return Ok(());
        }
        let symbol = self.symbol("hipMemset", "cudaMemset");
        let function: Symbol<'_, unsafe extern "C" fn(*mut c_void, c_int, usize) -> c_int> =
            unsafe { self.load_symbol(symbol)? };
        self.check("zero device memory", unsafe {
            function(destination, 0, byte_len)
        })
    }

    pub(super) fn copy_device_to_host(
        &self,
        source: *const c_void,
        output: &mut [u8],
    ) -> Result<(), MemError> {
        if output.is_empty() {
            return Ok(());
        }
        self.copy(
            output.as_mut_ptr().cast(),
            source,
            output.len(),
            MEMCPY_DEVICE_TO_HOST,
            "copy device to host",
        )
    }

    pub(crate) fn copy_device_to_device(
        &self,
        destination: *mut c_void,
        source: *const c_void,
        byte_len: usize,
    ) -> Result<(), MemError> {
        if byte_len == 0 {
            return Ok(());
        }
        self.copy(
            destination,
            source,
            byte_len,
            MEMCPY_DEVICE_TO_DEVICE,
            "copy device memory",
        )
    }

    pub(super) fn free(&self, data: *mut c_void) -> Result<(), MemError> {
        if data.is_null() {
            return Ok(());
        }
        let symbol = self.symbol("hipFree", "cudaFree");
        let function: Symbol<'_, unsafe extern "C" fn(*mut c_void) -> c_int> =
            unsafe { self.load_symbol(symbol)? };
        self.check("free device memory", unsafe { function(data) })
    }

    pub(crate) fn address_range(
        &self,
        data: *const c_void,
    ) -> Result<(*mut c_void, usize), MemError> {
        match self.dialect {
            GpuDialect::Hip => {
                let mut base = std::ptr::null_mut();
                let mut byte_len = 0;
                let function: Symbol<
                    '_,
                    unsafe extern "C" fn(*mut *mut c_void, *mut usize, *const c_void) -> c_int,
                > = unsafe { self.load_symbol("hipMemGetAddressRange")? };
                self.check("query device memory address range", unsafe {
                    function(&mut base, &mut byte_len, data)
                })?;
                Ok((base, byte_len))
            }
            GpuDialect::Cuda => {
                let mut base = 0_u64;
                let mut byte_len = 0;
                let library = self
                    .cuda_driver_library
                    .as_ref()
                    .expect("CUDA GPU API should have loaded the driver library");
                // Modern CUDA headers map cuMemGetAddressRange to this 64-bit ABI symbol.
                let function: Symbol<'_, unsafe extern "C" fn(*mut u64, *mut usize, u64) -> c_int> =
                    unsafe { self.load_symbol_from(library, "cuMemGetAddressRange_v2")? };
                self.check("query device memory address range", unsafe {
                    function(&mut base, &mut byte_len, data as usize as u64)
                })?;
                Ok((base as usize as *mut c_void, byte_len))
            }
        }
    }

    pub(crate) fn ipc_get_mem_handle(
        &self,
        data: *mut c_void,
    ) -> Result<RawIpcMemHandle, MemError> {
        let mut handle = RawIpcMemHandle {
            bytes: [0; IPC_HANDLE_BYTES],
        };
        let symbol = self.symbol("hipIpcGetMemHandle", "cudaIpcGetMemHandle");
        let function: Symbol<'_, unsafe extern "C" fn(*mut RawIpcMemHandle, *mut c_void) -> c_int> =
            unsafe { self.load_symbol(symbol)? };
        self.check("export IPC memory handle", unsafe {
            function(&mut handle, data)
        })?;
        Ok(handle)
    }

    pub(crate) fn ipc_open_mem_handle(
        &self,
        handle: RawIpcMemHandle,
    ) -> Result<*mut c_void, MemError> {
        let mut data = std::ptr::null_mut();
        let symbol = self.symbol("hipIpcOpenMemHandle", "cudaIpcOpenMemHandle");
        let function: Symbol<
            '_,
            unsafe extern "C" fn(*mut *mut c_void, RawIpcMemHandle, c_uint) -> c_int,
        > = unsafe { self.load_symbol(symbol)? };
        self.check("open IPC memory handle", unsafe {
            function(&mut data, handle, IPC_LAZY_ENABLE_PEER_ACCESS)
        })?;
        Ok(data)
    }

    pub(crate) fn ipc_close_mem_handle(&self, data: *mut c_void) -> Result<(), MemError> {
        if data.is_null() {
            return Ok(());
        }
        let symbol = self.symbol("hipIpcCloseMemHandle", "cudaIpcCloseMemHandle");
        let function: Symbol<'_, unsafe extern "C" fn(*mut c_void) -> c_int> =
            unsafe { self.load_symbol(symbol)? };
        self.check("close IPC memory handle", unsafe { function(data) })
    }

    pub(crate) fn synchronize(&self) -> Result<(), MemError> {
        let symbol = self.symbol("hipDeviceSynchronize", "cudaDeviceSynchronize");
        let function: Symbol<'_, unsafe extern "C" fn() -> c_int> =
            unsafe { self.load_symbol(symbol)? };
        self.check("synchronize device", unsafe { function() })
    }

    fn copy(
        &self,
        destination: *mut c_void,
        source: *const c_void,
        byte_len: usize,
        kind: c_int,
        operation: &'static str,
    ) -> Result<(), MemError> {
        let symbol = self.symbol("hipMemcpy", "cudaMemcpy");
        let function: Symbol<
            '_,
            unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int,
        > = unsafe { self.load_symbol(symbol)? };
        self.check(operation, unsafe {
            function(destination, source, byte_len, kind)
        })
    }

    fn symbol(&self, hip: &'static str, cuda: &'static str) -> &'static str {
        match self.dialect {
            GpuDialect::Hip => hip,
            GpuDialect::Cuda => cuda,
        }
    }

    unsafe fn load_symbol<T>(&self, symbol: &'static str) -> Result<Symbol<'_, T>, MemError> {
        unsafe { self.load_symbol_from(&self.runtime_library, symbol) }
    }

    unsafe fn load_symbol_from<'a, T>(
        &self,
        library: &'a Library,
        symbol: &'static str,
    ) -> Result<Symbol<'a, T>, MemError> {
        let symbol_cstr =
            CString::new(symbol).expect("runtime symbol names should not contain NUL");
        unsafe { library.get::<T>(symbol_cstr.as_bytes_with_nul()) }.map_err(|source| {
            MemError::LoadSymbol {
                dialect: self.dialect,
                symbol,
                source,
            }
        })
    }

    fn check(&self, operation: &'static str, status: c_int) -> Result<(), MemError> {
        if status == 0 {
            Ok(())
        } else {
            Err(MemError::GpuOperation {
                dialect: self.dialect,
                operation,
                status,
            })
        }
    }
}

fn load_library(dialect: GpuDialect, tried: Vec<PathBuf>) -> Result<Library, MemError> {
    let mut paths = tried.iter();
    let first = paths
        .next()
        .expect("GPU library candidate list should not be empty");
    let mut last_error = match unsafe { Library::new(first) } {
        Ok(library) => return Ok(library),
        Err(error) => error,
    };
    for path in paths {
        match unsafe { Library::new(path) } {
            Ok(library) => return Ok(library),
            Err(error) => last_error = error,
        }
    }
    Err(MemError::LoadLibrary {
        dialect,
        tried,
        source: last_error,
    })
}

/// Return a small, ordered set of runtime-library locations.
///
/// The bare soname delegates discovery to the platform loader. The remaining
/// paths come only from toolkit roots explicitly configured by the user or
/// development environment; this intentionally performs no filesystem search.
fn runtime_library_fallback_paths(dialect: GpuDialect) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    match dialect {
        GpuDialect::Hip => {
            for variable in ["ROCM_PATH", "HIP_PATH"] {
                if let Some(root) = env::var_os(variable) {
                    push_unique(&mut paths, PathBuf::from(root).join("lib/libamdhip64.so"));
                }
            }
        }
        GpuDialect::Cuda => {
            for variable in ["CUDA_PATH", "CUDA_HOME"] {
                if let Some(root) = env::var_os(variable) {
                    let root = PathBuf::from(root);
                    push_unique(&mut paths, root.join("lib64/libcudart.so"));
                    push_unique(&mut paths, root.join("lib/libcudart.so"));
                }
            }
        }
    }
    paths
}

fn push_unique(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !paths.contains(&path) {
        paths.push(path);
    }
}
