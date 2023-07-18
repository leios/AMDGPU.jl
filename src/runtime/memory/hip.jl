const _POOL_STATUS = AMDGPU.LockedObject(
    Dict{HIP.HIPDevice, Base.RefValue{Union{Nothing, Bool}}}())

function pool_status(dev::HIP.HIPDevice)
    Base.lock(_POOL_STATUS) do ps
        get!(ps, dev, Ref{Union{Nothing, Bool}}(nothing))
    end
end

const __pool_cleanup = Ref{Task}()
function pool_cleanup()
    idle_counters = Base.fill(0, HIP.ndevices())
    devices = HIP.devices()
    while true
        for (i, dev) in enumerate(devices)
            status = pool_status(dev)
            isnothing(status[]) && continue

            if status[]::Bool
                idle_counters[i] = 0
            else
                idle_counters[i] += 1
            end
            status[] = false

            if idle_counters[i] == 5
                old_device = HIP.device()
                old_device != dev && HIP.device!(dev)
                HIP.reclaim()
                old_device != dev && HIP.device!(old_device)
            end
        end

        try
            sleep(60)
        catch ex
            if ex isa EOFError
                # If we get EOF here, it's because Julia is shutting down,
                # so we should just exit the loop.
                break
            else
                rethrow()
            end
        end
    end
end

function mark_pool!(dev::HIP.HIPDevice)
    status = pool_status(dev)
    if isnothing(status[])
        # Default to `0` which is the default value in HIP.
        limit = parse_memory_limit(@load_preference("soft_memory_limit", "0 MiB"))
        HIP.attribute!(
            HIP.memory_pool(dev), HIP.hipMemPoolAttrReleaseThreshold, limit)
        if !isassigned(__pool_cleanup)
            __pool_cleanup[] = errormonitor(Threads.@spawn pool_cleanup())
        end
    end
    status[] = true
end

struct HIPBuffer <: AbstractAMDBuffer
    device::HIPDevice
    ptr::Ptr{Cvoid}
    bytesize::Int
    own::Bool
    _id::UInt64 # Unique ID used for refcounting.
end

function HIPBuffer(bytesize; stream::HIP.HIPStream)
    dev = stream.device
    bytesize == 0 && return HIPBuffer(dev, C_NULL, 0, true, _buffer_id!())

    mark_pool!(dev)
    pool = HIP.memory_pool(dev)

    has_limit = HARD_MEMORY_LIMIT != typemax(UInt64)

    ptr_ref = Ref{Ptr{Cvoid}}()
    alloc_or_retry!() do
        try
            # Try to ensure there is enough memory before even trying to allocate.
            if has_limit
                used = HIP.used_memory(pool)
                (used + bytesize) > HARD_MEMORY_LIMIT &&
                    throw(HIP.HIPError(HIP.hipErrorOutOfMemory))
            end

            # Try to allocate.
            HIP.hipMallocAsync(ptr_ref, bytesize, stream) |> HIP.check
            ptr_ref[] == C_NULL && throw(HIP.HIPError(HIP.hipErrorOutOfMemory))
            return HSA.STATUS_SUCCESS
        catch err
            # TODO rethrow if not out of memory error
            @debug "hipMallocAsync exception. Requested $(Base.format_bytes(bytesize))." exception=(err, catch_backtrace())
            return HSA.STATUS_ERROR_OUT_OF_RESOURCES
        end
    end
    ptr = ptr_ref[]
    @assert ptr != C_NULL "hipMallocAsync resulted in C_NULL for $(Base.format_bytes(bytesize))"

    # TODO ROCm 5.5+ has hard pool size limit
    if has_limit
        if HIP.reserved_memory(pool) > HARD_MEMORY_LIMIT
            HIP.reclaim() # TODO do not reclaim all memory
        end
        @assert HIP.reserved_memory(pool) ≤ HARD_MEMORY_LIMIT
    end

    HIPBuffer(dev, ptr, bytesize, true, _buffer_id!())
end

HIPBuffer(ptr::Ptr{Cvoid}, bytesize::Int) =
    HIPBuffer(AMDGPU.device(), ptr, bytesize, false, _buffer_id!())

Base.unsafe_convert(::Type{Ptr{T}}, buf::HIPBuffer) where T =
    convert(Ptr{T}, buf.ptr)

function view(buf::HIPBuffer, bytesize::Int)
    bytesize > buf.bytesize && throw(BoundsError(buf, bytesize))
    HIPBuffer(
        buf.device, buf.ptr + bytesize,
        buf.bytesize - bytesize, buf.own, buf._id)
end

function free(buf::HIPBuffer; stream::HIP.HIPStream)
    buf.own || return

    buf.ptr == C_NULL && return
    HIP.hipFreeAsync(buf, stream) |> HIP.check
    return
end

function upload!(dst::HIPBuffer, src::Ptr, bytesize::Int; stream::HIP.HIPStream)
    bytesize == 0 && return
    HIP.hipMemcpyHtoDAsync(dst, src, bytesize, stream) |> HIP.check
    return
end

function download!(dst::Ptr, src::HIPBuffer, bytesize::Int; stream::HIP.HIPStream)
    bytesize == 0 && return
    HIP.hipMemcpyDtoHAsync(dst, src, bytesize, stream) |> HIP.check
    return
end

function transfer!(dst::HIPBuffer, src::HIPBuffer, bytesize::Int; stream::HIP.HIPStream)
    bytesize == 0 && return
    HIP.hipMemcpyDtoDAsync(dst, src, bytesize, stream) |> HIP.check
    return
end

struct HostBuffer <: AbstractAMDBuffer
    device::HIPDevice
    ptr::Ptr{Cvoid}
    dev_ptr::Ptr{Cvoid}
    bytesize::Int
    own::Bool
    _id::UInt64 # Unique ID used for refcounting.
end

HostBuffer() = HostBuffer(AMDGPU.device(), C_NULL, C_NULL, 0, true, _buffer_id!())

function HostBuffer(bytesize::Integer, flags = 0)
    bytesize == 0 && return HostBuffer()

    ptr_ref = Ref{Ptr{Cvoid}}()
    HIP.hipHostMalloc(ptr_ref, bytesize, flags) |> HIP.check
    ptr = ptr_ref[]
    dev_ptr = get_device_ptr(ptr)
    HostBuffer(AMDGPU.device(), ptr, dev_ptr, bytesize, true, _buffer_id!())
end

function HostBuffer(ptr::Ptr{Cvoid}, sz::Integer)
    HIP.hipHostRegister(ptr, sz, HIP.hipHostRegisterMapped) |> HIP.check
    dev_ptr = get_device_ptr(ptr)
    HostBuffer(AMDGPU.device(), ptr, dev_ptr, sz, false, _buffer_id!())
end

function view(buf::HostBuffer, bytesize::Int)
    bytesize > buf.bytesize && throw(BoundsError(buf, bytesize))
    HostBuffer(
        buf.device,
        buf.ptr + bytesize, buf.dev_ptr + bytesize,
        buf.bytesize - bytesize, buf.own, buf._id)
end

upload!(dst::HostBuffer, src::Ptr, sz::Int; stream::HIP.HIPStream) =
    HIP.memcpy(dst, src, sz, HIP.hipMemcpyHostToHost, stream)

upload!(dst::HostBuffer, src::HIPBuffer, sz::Int; stream::HIP.HIPStream) =
    HIP.memcpy(dst, src, sz, HIP.hipMemcpyDeviceToHost, stream)

download!(dst::Ptr, src::HostBuffer, sz::Int; stream::HIP.HIPStream) =
    HIP.memcpy(dst, src, sz, HIP.hipMemcpyHostToHost, stream)

download!(dst::HIPBuffer, src::HostBuffer, sz::Int; stream::HIP.HIPStream) =
    HIP.memcpy(dst, src, sz, HIP.hipMemcpyHostToDevice, stream)

transfer!(dst::HostBuffer, src::HostBuffer, sz::Int; stream::HIP.HIPStream) =
    HIP.memcpy(dst, src, sz, HIP.hipMemcpyHostToHost, stream)

# download!(::Ptr, ::HIPBuffer)
transfer!(dst::HostBuffer, src::HIPBuffer, sz::Int; stream::HIP.HIPStream) =
    HIP.memcpy(dst, src, sz, HIP.hipMemcpyDeviceToHost, stream)

# upload!(::HIPBuffer, ::Ptr)
transfer!(dst::HIPBuffer, src::HostBuffer, sz::Int; stream::HIP.HIPStream) =
    HIP.memcpy(dst, src, sz, HIP.hipMemcpyHostToDevice, stream)

Base.unsafe_convert(::Type{Ptr{T}}, buf::HostBuffer) where T =
    convert(Ptr{T}, buf.ptr)

@inline device_ptr(buf::HostBuffer) = buf.dev_ptr

function free(buf::HostBuffer; kwargs...)
    buf.ptr == C_NULL && return
    if buf.own
        HIP.hipHostFree(buf) |> HIP.check
    else
        is_pinned(buf.dev_ptr) && HIP.check(HIP.hipHostUnregister(buf.ptr))
    end
    return
end

function get_device_ptr(ptr::Ptr{Cvoid})
    ptr == C_NULL && return C_NULL
    ptr_ref = Ref{Ptr{Cvoid}}()
    HIP.hipHostGetDevicePointer(ptr_ref, ptr, 0) |> HIP.check
    ptr_ref[]
end

function is_pinned(ptr::Ptr{Cvoid})
    ptr == C_NULL && return false

    st, data = attributes(ptr)
    if st == HIP.hipErrorInvalidValue
        return false
    elseif st == HIP.hipSuccess
        return data.memoryType == HIP.hipMemoryTypeHost
    end
    st |> HIP.check
end

function attributes(ptr::Ptr{Cvoid})
    data = Ref{HIP.hipPointerAttribute_t}()
    st = HIP.hipPointerGetAttributes(data, ptr)
    st, data[]
end