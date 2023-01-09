# HACK: callback function for `launch_configuration` on platforms without support for
#       trampolines as used by `@cfunction` (JuliaLang/julia#27174, JuliaLang/julia#32154)
_localmem_cb = nothing
_localmem_cint_cb(x::Cint) = Cint(something(_localmem_cb)(x))
_localmem_cb_lock = Threads.ReentrantLock()

"""
    launch_configuration(fun::ROCFunction; localmem=0, max_threads=0)

Calculate a suggested launch configuration for kernel `fun` requiring `localmem` bytes of
dynamic shared memory. Returns a tuple with a suggested amount of threads, and the minimal
amount of blocks to reach maximal occupancy. Optionally, the maximum amount of threads can
be constrained using `max_threads`.

In the case of a variable amount of shared memory, pass a callable object for `localmem`
instead, taking a single integer representing the block size and returning the amount of
dynamic shared memory for that configuration.
"""
function launch_configuration(fun::Runtime.ROCFunction; localmem::Union{Integer,Base.Callable}=0,
                              max_threads::Integer=0)
    blocks_ref = Ref{Cint}()
    threads_ref = Ref{Cint}()
    if isa(localmem, Integer)
        cuOccupancyMaxPotentialBlockSize(blocks_ref, threads_ref, fun, C_NULL, localmem, max_threads)
    elseif Sys.ARCH == :x86 || Sys.ARCH == :x86_64
        localmem_cint = threads -> Cint(localmem(threads))
        cb = @cfunction($localmem_cint, Cint, (Cint,))
        cuOccupancyMaxPotentialBlockSize(blocks_ref, threads_ref, fun, cb, 0, max_threads)
    else
        lock(_localmem_cb_lock) do
            global _localmem_cb
            _localmem_cb = localmem
            cb = @cfunction(_localmem_cint_cb, Cint, (Cint,))
            cuOccupancyMaxPotentialBlockSize(blocks_ref, threads_ref, fun, cb, 0, max_threads)
            _localmem_cb = nothing
        end
    end
    return (blocks=Int(blocks_ref[]), threads=Int(threads_ref[]))
end

# scan and accumulate (shamelessly stolen from CUDA.jl: https://github.com/JuliaGPU/CUDA.jl/blob/master/src/accumulate.jl)

## COV_EXCL_START

# partial scan of individual thread blocks within a grid
# work-efficient implementation after Blelloch (1990)
#
# number of threads needs to be a power-of-2
#
# performance TODOs:
# - shuffle
# - warp-aggregate atomics
# - the ND case is quite a bit slower than the 1D case (not using Cartesian indices,
function partial_scan(op::Function, output::AbstractArray{T}, input::AbstractArray,
                      Rdim, Rpre, Rpost, Rother, neutral, init,
                      ::Val{inclusive}=Val(true)) where {T, inclusive}
#=
    threads = workgroupDim().x
    thread = workitemIdx().x
    block = workgroupIdx().x

    temp = ROCDeviceArray(T, (2*threads,))

    # iterate the main dimension using threads and the first block dimension
    i = (workgroupIdx().x-1i32) * workgroupDim().x + workitemIdx().x
    # iterate the other dimensions using the remaining block dimensions
    j = (workgroupIdx().z-1i32) * gridDim().y + workgroupIdx().y

    if j > length(Rother)
        return
    end

    @inbounds begin
        I = Rother[j]
        Ipre = Rpre[I[1]]
        Ipost = Rpost[I[2]]
    end

    # load input into shared memory (apply `op` to have the correct type)
    @inbounds temp[thread] = if i <= length(Rdim)
        op(neutral, input[Ipre, i, Ipost])
    else
        op(neutral, neutral)
    end

    # build sum in place up the tree
    offset = 1
    d = threads>>1
    while d > 0
        sync_workgroup()
        @inbounds if thread <= d
            ai = offset * (2*thread-1)
            bi = offset * (2*thread)
            temp[bi] = op(temp[ai], temp[bi])
        end
        offset *= 2
        d >>= 1
    end

    # clear the last element
    @inbounds if thread == 1
        temp[threads] = neutral
    end

    # traverse down tree & build scan
    d = 1
    while d < threads
        offset >>= 1
        sync_workgroup()
        @inbounds if thread <= d
            ai = offset * (2*thread-1)
            bi = offset * (2*thread)

            t = temp[ai]
            temp[ai] = temp[bi]
            temp[bi] = op(t, temp[bi])
        end
        d *= 2
    end

    sync_workgroup()

    # write results to device memory
    @inbounds if i <= length(Rdim)
        val = if inclusive
            op(temp[thread], input[Ipre, i, Ipost])
        else
            temp[thread]
        end
        if init !== nothing
            val = op(something(init), val)
        end
        output[Ipre, i, Ipost] = val
    end
=#

    return
end

# aggregate the result of a partial scan by applying preceding block aggregates
function aggregate_partial_scan(op::Function, output::AbstractArray,
                                aggregates::AbstractArray, Rdim, Rpre, Rpost, Rother,
                                init)
#=
    threads = workgroupDim().x
    thread = workitemIdx().x
    block = workgroupIdx().x

    # iterate the main dimension using threads and the first block dimension
    i = (workgroupIdx().x-1i32) * workgroupDim().x + workitemIdx().x
    # iterate the other dimensions using the remaining block dimensions
    j = (workgroupIdx().z-1i32) * gridDim().y + workgroupIdx().y

    @inbounds if i <= length(Rdim) && j <= length(Rother)
        I = Rother[j]
        Ipre = Rpre[I[1]]
        Ipost = Rpost[I[2]]

        val = if block > 1
            op(aggregates[Ipre, block-1, Ipost], output[Ipre, i, Ipost])
        else
            output[Ipre, i, Ipost]
        end

        if init !== nothing
            val = op(something(init), val)
        end

        output[Ipre, i, Ipost] = val
    end
=#

    return
end

## COV_EXCL_STOP

function scan!(f::Function, output::AnyROCArray{T}, input::AnyROCArray;
               dims::Integer, init=nothing, neutral=GPUArrays.neutral_element(f, T)) where {T}
    dims > 0 || throw(ArgumentError("dims must be a positive integer"))
    inds_t = axes(input)
    axes(output) == inds_t || throw(DimensionMismatch("shape of B must match A"))
    dims > ndims(input) && return copyto!(output, input)
    isempty(inds_t[dims]) && return output

    # iteration domain across the main dimension
    Rdim = CartesianIndices((size(input, dims),))

    # iteration domain for the other dimensions
    Rpre = CartesianIndices(size(input)[1:dims-1])
    Rpost = CartesianIndices(size(input)[dims+1:end])
    Rother = CartesianIndices((length(Rpre), length(Rpost)))

    # determine how many threads we can launch for the scan kernel
    kernel = @roc launch=false partial_scan(f, output, input, Rdim, Rpre, Rpost, Rother, neutral, init, Val(true))
    kernel_config = launch_configuration(kernel.fun; localmem=(threads)->2*threads*sizeof(T))

    # determine the grid layout to cover the other dimensions
    if length(Rother) > 1
        dev = device()
        max_other_blocks = attribute(dev, DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)
        blocks_other = (min(length(Rother), max_other_blocks),
                        cld(length(Rother), max_other_blocks))
    else
        blocks_other = (1, 1)
    end

    # does that suffice to scan the array in one go?
    full = nextpow(2, length(Rdim))
    if full <= kernel_config.threads
        @roc(threads=full, blocks=(1, blocks_other...), localmem=2*full*sizeof(T), name="scan",
              partial_scan(f, output, input, Rdim, Rpre, Rpost, Rother, neutral, init, Val(true)))
    else
        # perform partial scans across the scanning dimension
        # NOTE: don't set init here to avoid applying the value multiple times
        partial = prevpow(2, kernel_config.threads)
        blocks_dim = cld(length(Rdim), partial)
        @roc(threads=partial, blocks=(blocks_dim, blocks_other...), localmem=2*partial*sizeof(T),
              partial_scan(f, output, input, Rdim, Rpre, Rpost, Rother, neutral, nothing, Val(true)))

        # get the total of each thread block (except the first) of the partial scans
        aggregates = fill(neutral, Base.setindex(size(input), blocks_dim, dims))
        copyto!(aggregates, selectdim(output, dims, partial:partial:length(Rdim)))

        # scan these totals to get totals for the entire partial scan
        accumulate!(f, aggregates, aggregates; dims=dims)

        # add those totals to the partial scan result
        # NOTE: we assume that this kernel requires fewer resources than the scan kernel.
        #       if that does not hold, launch with fewer threads and calculate
        #       the aggregate block index within the kernel itself.
        @roc(threads=partial, blocks=(blocks_dim, blocks_other...),
              aggregate_partial_scan(f, output, aggregates, Rdim, Rpre, Rpost, Rother, init))

        unsafe_free!(aggregates)
    end

    return output
end


## Base interface

Base._accumulate!(op, output::AnyROCArray, input::AnyROCVector, dims::Nothing, init::Nothing) =
    scan!(op, output, input; dims=1)

Base._accumulate!(op, output::AnyROCArray, input::AnyROCArray, dims::Integer, init::Nothing) =
    scan!(op, output, input; dims=dims)

Base._accumulate!(op, output::AnyROCArray, input::ROCVector, dims::Nothing, init::Some) =
    scan!(op, output, input; dims=1, init=init)

Base._accumulate!(op, output::AnyROCArray, input::AnyROCArray, dims::Integer, init::Some) =
    scan!(op, output, input; dims=dims, init=init)

Base.accumulate_pairwise!(op, result::AnyROCVector, v::AnyROCVector) = accumulate!(op, result, v)
