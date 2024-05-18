struct HIPCompilerParams <: AbstractCompilerParams
    # Whether to compile kernel for the wavefront of size 64.
    wavefrontsize64::Bool
    # AMD GPU devices support fast atomic read-modify-write (RMW)
    # operations on floating-point values.
    # On single- or double-precision floating-point values this may generate
    # a hardware RMW instruction that is faster than emulating
    # the atomic operation using an atomic compare-and-swap (CAS) loop.
    unsafe_fp_atomics::Bool
end

const HIPCompilerConfig = CompilerConfig{GCNCompilerTarget, HIPCompilerParams}
const HIPCompilerJob = CompilerJob{GCNCompilerTarget, HIPCompilerParams}

const _hip_compiler_cache = Dict{HIP.HIPDevice, Dict{Any, HIP.HIPFunction}}()

# hash(fun, hash(f, hash(tt))) => HIPKernel
const _kernel_instances = Dict{UInt, Runtime.HIPKernel}()

function compiler_cache(dev::HIP.HIPDevice)
    get!(() -> Dict{UInt, Any}(), _hip_compiler_cache, dev)
end

GPUCompiler.runtime_module(@nospecialize(::HIPCompilerJob)) = AMDGPU

GPUCompiler.method_table(@nospecialize(::HIPCompilerJob)) = AMDGPU.method_table

GPUCompiler.kernel_state_type(@nospecialize(::HIPCompilerJob)) = AMDGPU.KernelState

function GPUCompiler.link_libraries!(
    @nospecialize(job::HIPCompilerJob), mod::LLVM.Module,
    undefined_fns::Vector{String},
)
    invoke(GPUCompiler.link_libraries!,
        Tuple{CompilerJob{GCNCompilerTarget}, typeof(mod), typeof(undefined_fns)},
        job, mod, undefined_fns)
    link_device_libs!(
        job.config.target, mod, undefined_fns;
        wavefrontsize64=job.config.params.wavefrontsize64)
end

# FIXME this shouldn't be needed
function GPUCompiler.finish_ir!(
    @nospecialize(job::HIPCompilerJob), mod::LLVM.Module, entry::LLVM.Function,
)
    undefined_fns = GPUCompiler.decls(mod)
    isempty(undefined_fns) && return entry
    link_device_libs!(
        job.config.target, mod, LLVM.name.(undefined_fns);
        wavefrontsize64=job.config.params.wavefrontsize64)
    return entry
end

function GPUCompiler.finish_module!(
    @nospecialize(job::HIPCompilerJob), mod::LLVM.Module, entry::LLVM.Function,
)
    entry = invoke(GPUCompiler.finish_module!,
        Tuple{CompilerJob{GCNCompilerTarget}, typeof(mod), typeof(entry)},
        job, mod, entry)

    # Workaround for the lack of zeroinitializer support for LDS.
    zeroinit_lds!(mod, entry)

    # Force-inline exception-related functions.
    # LLVM gets confused when not all functions are inlined,
    # causing huge scratch memory usage.
    # And GPUCompiler fails to inline all functions without forcing
    # always-inline attributes on them. Add them here.
    target_fns = (
        "signal_exception", "report_exception", "malloc", "__throw_")
    inline_attr = EnumAttribute("alwaysinline")
    atomic_attr = StringAttribute("amdgpu-unsafe-fp-atomics", "true")

    for fn in LLVM.functions(mod)
        do_inline = any(occursin.(target_fns, LLVM.name(fn)))
        if job.config.params.unsafe_fp_atomics || do_inline
            attrs = LLVM.function_attributes(fn)

            do_inline && inline_attr ∉ collect(attrs) &&
                push!(attrs, inline_attr)
            job.config.params.unsafe_fp_atomics &&
                push!(attrs, atomic_attr)
        end
    end

    return entry
end

function parse_llvm_features(arch::String)
    splits = split(arch, ":")
    length(splits) == 1 && return (; dev_isa=splits[1], features="")

    dev_isa, features = splits[1], splits[2:end]
    features = join(map(x -> x[1:end - 1], filter(x -> x[end] == '+', features)), ",+")
    isempty(features) || (features = "+" * features)
    (; dev_isa, features)
end


function compiler_config(dev::HIP.HIPDevice;
    name::Union{String, Nothing} = nothing, kernel::Bool = true,
    unsafe_fp_atomics::Bool = true,
)
    dev_isa, features = parse_llvm_features(HIP.gcn_arch(dev))
    target = GCNCompilerTarget(; dev_isa, features)
    params = HIPCompilerParams(HIP.wavefrontsize(dev) == 64, unsafe_fp_atomics)
    CompilerConfig(target, params; kernel, name, always_inline=true)
end

const hipfunction_lock = ReentrantLock()

"""
    hipfunction(f::F, tt::TT = Tuple{}; kwargs...)

Compile Julia function `f` to a HIP kernel given a tuple of
argument's types `tt` that it accepts.

The following kwargs are supported:

- `name::Union{String, Nothing} = nothing`:
    A unique name to give a compiled kernel.
- `unsafe_fp_atomics::Bool = true`:
    Whether to use 'unsafe' floating-point atomics.
    AMD GPU devices support fast atomic read-modify-write (RMW)
    operations on floating-point values.
    On single- or double-precision floating-point values this may generate
    a hardware RMW instruction that is faster than emulating
    the atomic operation using an atomic compare-and-swap (CAS) loop.
"""
function hipfunction(f::F, tt::TT = Tuple{}; kwargs...) where {F <: Core.Function, TT}
    Base.@lock hipfunction_lock begin
        dev = AMDGPU.device()
        cache = compiler_cache(dev)
        config = compiler_config(dev; kwargs...)

        source = methodinstance(F, tt)
        fun = GPUCompiler.cached_compilation(
            cache, source, config, hipcompile, hiplink)

        h = hash(fun, hash(f, hash(tt)))
        kernel = get!(_kernel_instances, h) do
            Runtime.HIPKernel{F, tt}(f, fun)
        end
        return kernel::Runtime.HIPKernel{F, tt}
    end
end

function create_executable(obj)
    lld = if AMDGPU.lld_artifact
        `$(LLD_jll.lld()) -flavor gnu`
    else
        @assert !isempty(AMDGPU.lld_path) "ld.lld was not found; cannot link kernel"
        `$(AMDGPU.lld_path)`
    end

    path_o = tempname(;cleanup=false) * ".obj"
    path_exe = tempname(;cleanup=false) * ".exe"

    write(path_o, obj)
    run(`$lld -shared -o $path_exe $path_o`)
    bin = read(path_exe)

    rm(path_o)
    rm(path_exe)
    return bin
end

function hipcompile(@nospecialize(job::CompilerJob))
    obj, meta = JuliaContext() do ctx
        GPUCompiler.compile(:obj, job)
    end

    entry = LLVM.name(meta.entry)
    globals = filter(isextinit, collect(LLVM.globals(meta.ir))) .|> LLVM.name

    global_hostcall_names = (
        :malloc_hostcall, :free_hostcall, :print_hostcall, :printf_hostcall)
    global_hostcalls = Symbol[]
    for gbl in LLVM.globals(meta.ir), gbl_name in global_hostcall_names
        occursin("__$gbl_name", LLVM.name(gbl)) || continue
        push!(global_hostcalls, gbl_name)
    end
    if !isempty(global_hostcalls)
        @warn """Global hostcalls detected: $global_hostcalls.
        Use `AMDGPU.synchronize(; stop_hostcalls=false)` to synchronize and stop them.
        Otherwise, performance might degrade if they keep running in the background.
        """ maxlog=1
    end

    if !isempty(globals)
        @warn """
        HIP backend does not support setting extinit globals.
        But kernel `$entry` has following:
        $globals

        Compilation will likely fail.
        """
    end
    (; obj=create_executable(codeunits(obj)), entry, global_hostcalls)
end

function hiplink(@nospecialize(job::CompilerJob), compiled)
    (; obj, entry, global_hostcalls) = compiled
    mod = HIP.HIPModule(obj)
    HIP.HIPFunction(mod, entry, global_hostcalls)
end

function run_and_collect(cmd)
    stdout = Pipe()
    proc = run(pipeline(ignorestatus(cmd); stdout, stderr=stdout), wait=false)
    close(stdout.in)

    reader = Threads.@spawn String(read(stdout))
    Base.wait(proc)
    log = strip(fetch(reader))
    return proc, log
end
