//! POSIX VMM allocations. Exported descriptors confer allocation ownership;
//! READ access is a property of this process's mapping, not of the descriptor.
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

use super::*;

#[repr(C)]
#[derive(Clone, Copy)]
struct Location {
    kind: c_int,
    id: c_int,
}

#[repr(C)]
struct AllocationFlags {
    compression: u8,
    rdma: u8,
    usage: u16,
}

#[repr(C)]
struct HipProperties {
    kind: c_int,
    handle_types: c_int,
    location: Location,
    win32_metadata: *mut c_void,
    flags: AllocationFlags,
}

#[repr(C)]
struct CudaProperties {
    kind: c_int,
    handle_types: c_int,
    location: Location,
    win32_metadata: *mut c_void,
    flags: AllocationFlags,
    reserved: [u8; 4],
}

#[repr(C)]
struct AccessDesc {
    location: Location,
    flags: c_int,
}

// HIP uses pointers for both addresses and handles; CUDA uses uint64_t.
// Keep those signatures distinct, including the different property layouts.
struct Functions<A, P> {
    granularity: unsafe extern "C" fn(*mut usize, *const P, c_int) -> c_int,
    create: unsafe extern "C" fn(*mut A, usize, *const P, u64) -> c_int,
    import: unsafe extern "C" fn(*mut A, *mut c_void, c_int) -> c_int,
    export: unsafe extern "C" fn(*mut c_void, A, c_int, u64) -> c_int,
    reserve: unsafe extern "C" fn(*mut A, usize, usize, A, u64) -> c_int,
    map: unsafe extern "C" fn(A, usize, usize, A, u64) -> c_int,
    access: unsafe extern "C" fn(A, usize, *const AccessDesc, usize) -> c_int,
    unmap: unsafe extern "C" fn(A, usize) -> c_int,
    free_address: unsafe extern "C" fn(A, usize) -> c_int,
    release: unsafe extern "C" fn(A) -> c_int,
}

impl<A, P> Functions<A, P> {
    fn load(gpu: &GpuApi) -> Result<Self, MemError> {
        let library = match gpu.dialect {
            GpuDialect::Hip => &gpu.runtime_library,
            GpuDialect::Cuda => gpu
                .cuda_driver_library
                .as_ref()
                .expect("CUDA driver loaded"),
        };
        macro_rules! symbol {
            ($hip:literal, $cuda:literal) => {
                unsafe { *gpu.load_symbol_from(library, gpu.symbol($hip, $cuda))? }
            };
        }
        // Resolve every operation, including rollback, before creating resources.
        Ok(Self {
            granularity: symbol!(
                "hipMemGetAllocationGranularity",
                "cuMemGetAllocationGranularity"
            ),
            create: symbol!("hipMemCreate", "cuMemCreate"),
            import: symbol!(
                "hipMemImportFromShareableHandle",
                "cuMemImportFromShareableHandle"
            ),
            export: symbol!(
                "hipMemExportToShareableHandle",
                "cuMemExportToShareableHandle"
            ),
            reserve: symbol!("hipMemAddressReserve", "cuMemAddressReserve"),
            map: symbol!("hipMemMap", "cuMemMap"),
            access: symbol!("hipMemSetAccess", "cuMemSetAccess"),
            unmap: symbol!("hipMemUnmap", "cuMemUnmap"),
            free_address: symbol!("hipMemAddressFree", "cuMemAddressFree"),
            release: symbol!("hipMemRelease", "cuMemRelease"),
        })
    }
}

struct Mapping<A: Copy, P> {
    gpu: Arc<GpuApi>,
    functions: Functions<A, P>,
    handle: A,
    address: A,
    pointer: *mut c_void,
    allocation_len: usize,
    byte_len: usize,
    reserved: bool,
    mapped: bool,
}

/// Retains the allocation handle and its process-local GPU mapping.
/// The caller must finish all GPU work using `ptr()` before dropping it.
pub(crate) struct SharedAllocation {
    mapping: SharedMapping,
}

enum SharedMapping {
    Hip(Mapping<*mut c_void, HipProperties>),
    Cuda(Mapping<u64, CudaProperties>),
}

impl std::fmt::Debug for SharedAllocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedAllocation")
            .field("ptr", &self.ptr())
            .field("byte_len", &self.byte_len())
            .field("allocation_len", &self.allocation_len())
            .finish()
    }
}

impl SharedAllocation {
    pub(crate) fn ptr(&self) -> *mut c_void {
        match &self.mapping {
            SharedMapping::Hip(m) => m.pointer,
            SharedMapping::Cuda(m) => m.pointer,
        }
    }

    pub(crate) fn byte_len(&self) -> usize {
        match &self.mapping {
            SharedMapping::Hip(m) => m.byte_len,
            SharedMapping::Cuda(m) => m.byte_len,
        }
    }

    pub(crate) fn allocation_len(&self) -> usize {
        match &self.mapping {
            SharedMapping::Hip(m) => m.allocation_len,
            SharedMapping::Cuda(m) => m.allocation_len,
        }
    }

    /// Export a new descriptor owned by the caller. It permits another process
    /// to import the backing allocation; it does not restrict future mappings.
    pub(crate) fn export_fd(&self) -> Result<OwnedFd, MemError> {
        match &self.mapping {
            SharedMapping::Hip(m) => m.export_fd(),
            SharedMapping::Cuda(m) => m.export_fd(),
        }
    }
}

impl GpuApi {
    pub(crate) fn own_shared(self: &Arc<Self>, bytes: &[u8]) -> Result<SharedAllocation, MemError> {
        let allocation = self.shared_mapping(bytes.len(), None)?;
        self.zero(allocation.ptr(), allocation.allocation_len())?;
        self.copy_host_to_device(allocation.ptr(), bytes)?;
        self.synchronize()?;
        Ok(allocation)
    }

    pub(crate) fn import_shared(
        self: &Arc<Self>,
        fd: OwnedFd,
        byte_len: usize,
        allocation_len: usize,
    ) -> Result<SharedAllocation, MemError> {
        self.shared_mapping(byte_len, Some((fd, allocation_len)))
    }

    fn shared_mapping(
        self: &Arc<Self>,
        byte_len: usize,
        imported: Option<(OwnedFd, usize)>,
    ) -> Result<SharedAllocation, MemError> {
        if byte_len == 0 {
            return Err(invalid("shared allocation must not be empty"));
        }
        if self.dialect == GpuDialect::Hip {
            let version: Symbol<'_, unsafe extern "C" fn(*mut c_int) -> c_int> =
                unsafe { self.load_symbol("hipRuntimeGetVersion")? };
            let mut runtime_version = 0;
            self.check("query HIP VMM runtime version", unsafe {
                version(&mut runtime_version)
            })?;
            // HIP encodes major*10_000_000 + minor*100_000 + patch.
            // Older ROCm VMM paths can report READ while installing writable
            // GPU mappings. Require the runtime validated with hardware READ.
            if runtime_version < 71_500_000 {
                return Err(invalid(
                    "shared GPU allocations require HIP 7.15 or newer with enforced VMM read-only access",
                ));
            }
        }
        // Runtime device selection and cudaFree(NULL) establish the same primary
        // context used by subsequent runtime copies and generated kernels, also
        // when this cached GpuApi is used on a different host thread.
        let select: Symbol<'_, unsafe extern "C" fn(c_int) -> c_int> =
            unsafe { self.load_symbol(self.symbol("hipSetDevice", "cudaSetDevice"))? };
        self.check("select VMM device", unsafe { select(0) })?;
        let initialize: Symbol<'_, unsafe extern "C" fn(*mut c_void) -> c_int> =
            unsafe { self.load_symbol(self.symbol("hipFree", "cudaFree"))? };
        self.check("initialize VMM context", unsafe {
            initialize(std::ptr::null_mut())
        })?;
        let location = Location { kind: 1, id: 0 };
        let flags = AllocationFlags {
            compression: 0,
            rdma: 0,
            usage: 0,
        };
        let mapping = match self.dialect {
            GpuDialect::Hip => {
                let properties = HipProperties {
                    kind: 1,
                    handle_types: 1,
                    location,
                    win32_metadata: std::ptr::null_mut(),
                    flags,
                };
                SharedMapping::Hip(Mapping::new(
                    self.clone(),
                    &properties,
                    location,
                    byte_len,
                    imported,
                    std::ptr::null_mut(),
                    |p| p,
                )?)
            }
            GpuDialect::Cuda => {
                let properties = CudaProperties {
                    kind: 1,
                    handle_types: 1,
                    location,
                    win32_metadata: std::ptr::null_mut(),
                    flags,
                    reserved: [0; 4],
                };
                SharedMapping::Cuda(Mapping::new(
                    self.clone(),
                    &properties,
                    location,
                    byte_len,
                    imported,
                    0,
                    |p| p as usize as *mut c_void,
                )?)
            }
        };
        Ok(SharedAllocation { mapping })
    }
}

fn invalid(reason: &'static str) -> MemError {
    MemError::InvalidSharedAllocation { reason }
}

impl<A: Copy, P> Mapping<A, P> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        gpu: Arc<GpuApi>,
        properties: &P,
        location: Location,
        byte_len: usize,
        imported: Option<(OwnedFd, usize)>,
        zero: A,
        pointer: fn(A) -> *mut c_void,
    ) -> Result<Self, MemError> {
        let functions = Functions::load(&gpu)?;
        let mut granularity = 0;
        gpu.check("query VMM allocation granularity", unsafe {
            (functions.granularity)(&mut granularity, properties, 0)
        })?;
        if granularity == 0 {
            return Err(invalid("GPU returned zero VMM granularity"));
        }
        let minimum_len = byte_len
            .checked_add(granularity - 1)
            .map(|n| n / granularity * granularity)
            .ok_or(MemError::AllocationSizeOverflow {
                element_count: byte_len,
                element_size: granularity,
            })?;
        let allocation_len = imported.as_ref().map_or(minimum_len, |(_, len)| *len);
        if allocation_len < byte_len || allocation_len % granularity != 0 {
            return Err(invalid(
                "import length must cover the data and be a multiple of VMM granularity",
            ));
        }
        let read_only = imported.is_some();
        let mut handle = zero;
        match imported {
            Some((fd, _)) => gpu.check("import shared VMM allocation", unsafe {
                // Both APIs take the numeric POSIX fd cast to void*, not &fd.
                (functions.import)(&mut handle, fd.as_raw_fd() as usize as *mut c_void, 1)
            })?,
            None => gpu.check("create shared VMM allocation", unsafe {
                (functions.create)(&mut handle, allocation_len, properties, 0)
            })?,
        }
        let mut allocation = Self {
            gpu,
            functions,
            handle,
            address: zero,
            pointer: std::ptr::null_mut(),
            allocation_len,
            byte_len,
            reserved: false,
            mapped: false,
        };
        allocation.gpu.check("reserve VMM address", unsafe {
            (allocation.functions.reserve)(&mut allocation.address, allocation_len, 0, zero, 0)
        })?;
        allocation.reserved = true;
        allocation.pointer = pointer(allocation.address);
        if allocation.pointer.is_null() {
            return Err(invalid("GPU returned a null VMM address"));
        }
        allocation.gpu.check("map shared VMM allocation", unsafe {
            (allocation.functions.map)(allocation.address, allocation_len, 0, handle, 0)
        })?;
        allocation.mapped = true;
        let access = AccessDesc {
            location,
            flags: if read_only { 1 } else { 3 },
        };
        allocation.gpu.check("set shared VMM access", unsafe {
            (allocation.functions.access)(allocation.address, allocation_len, &access, 1)
        })?;
        Ok(allocation)
    }

    fn export_fd(&self) -> Result<OwnedFd, MemError> {
        let mut fd: c_int = -1;
        self.gpu.check("export shared VMM allocation", unsafe {
            (self.functions.export)((&mut fd as *mut c_int).cast(), self.handle, 1, 0)
        })?;
        if fd < 0 {
            return Err(invalid(
                "GPU returned an invalid shared allocation descriptor",
            ));
        }
        // SAFETY: success returns a fresh descriptor owned by the exporter.
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        // Prevent accidental inheritance through exec. The vendor export API
        // cannot set CLOEXEC atomically, so export must not race a process exec.
        if unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) } < 0 {
            return Err(MemError::SharedHandleIo(std::io::Error::last_os_error()));
        }
        Ok(fd)
    }
}

impl<A: Copy, P> Drop for Mapping<A, P> {
    fn drop(&mut self) {
        // Failure to unmap must not be followed by freeing an address range
        // that might still be accessible to GPU work.
        let check = |operation, status| {
            if let Err(error) = self.gpu.check(operation, status) {
                eprintln!("fatal GPU mapping cleanup failure: {error}");
                std::process::abort();
            }
        };
        unsafe {
            if self.mapped {
                check(
                    "unmap shared VMM allocation",
                    (self.functions.unmap)(self.address, self.allocation_len),
                );
            }
            if self.reserved {
                check(
                    "free VMM address",
                    (self.functions.free_address)(self.address, self.allocation_len),
                );
            }
            check(
                "release shared VMM allocation",
                (self.functions.release)(self.handle),
            );
        }
    }
}
