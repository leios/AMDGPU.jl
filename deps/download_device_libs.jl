## Load ROCm Device-Libs

using BinaryProvider # requires BinaryProvider 0.3.0 or later

# Parse some basic command-line arguments
const verbose = "--verbose" in ARGS
const prefix = Prefix(get([a for a in ARGS if a != "--verbose"], 1, joinpath(@__DIR__, "usr")))
products = [
    FileProduct(prefix, "lib/ockl.amdgcn.bc", :rocmdevlibdir),
]

# Download binaries from hosted location
bin_prefix = "https://github.com/jpsamaroo/ROCmDeviceLibsDownloader/releases/download/v2.2.0"

# Listing of files generated by BinaryBuilder:
download_info = Dict(
    Linux(:x86_64, libc=:glibc) => ("$bin_prefix/ROCmDeviceLibsDownloader.v2.2.0.x86_64-linux-gnu.tar.gz", "4bd7c9aaa56f7e72d8707939b28106162df75786ab7a30c35622220aa3a4b7db"),
    Linux(:x86_64, libc=:musl) => ("$bin_prefix/ROCmDeviceLibsDownloader.v2.2.0.x86_64-linux-musl.tar.gz", "dafd049ddeb76491f85bbe7897b4e004326bb8af76e639e925748cad05391a20"),
)

# Install unsatisfied or updated dependencies:
unsatisfied = any(!satisfied(p; verbose=verbose) for p in products)
dl_info = choose_download(download_info, platform_key_abi())
if dl_info === nothing && unsatisfied
    # If we don't have a compatible .tar.gz to download, complain.
    # Alternatively, you could attempt to install from a separate provider,
    # build from source or something even more ambitious here.
    @warn "Your platform (\"$(Sys.MACHINE)\", parsed as \"$(triplet(platform_key_abi()))\") is not supported by this package!"
    exit(0)
end

# If we have a download, and we are unsatisfied (or the version we're
# trying to install is not itself installed) then load it up!
if unsatisfied || !isinstalled(dl_info...; prefix=prefix)
    # Download and install binaries
    install(dl_info...; prefix=prefix, force=true, verbose=verbose)
end

# Write out a deps.jl file that will contain mappings for our products
write_deps_file(joinpath(@__DIR__, "deps.jl"), products, verbose=verbose)