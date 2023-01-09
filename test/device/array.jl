@testset "ROCDeviceArray" begin
    RA = ROCArray(rand(4,4))
    RD = rocconvert(RA)
    @test RD isa AMDGPU.Device.ROCDeviceMatrix

    # Host indexing is disallowed
    @test_throws Exception RD[1]
    @test_throws Exception RD[1,1]
    RD_view = @view RD[2:3, 2:3]
    @test RD_view isa SubArray
    @test_throws Exception RD_view[1]
    @test_throws Exception RD_view[1,1]
    RD_adj = RD'
    @test RD_adj isa LinearAlgebra.Adjoint
    @test_throws Exception RD_adj[1]
    @test_throws Exception RD_adj[1,1]

    # Custom show methods are defined
    @test occursin("4×4 device array at", sprint(io->show(io, RD)))
    @test occursin("2×2 device array view", sprint(io->show(io, RD_view)))
    @test occursin("4×4 device array wrapper Adjoint", sprint(io->show(io, RD_adj)))

    # Custom hash methods are defined
    @test AMDGPU.Runtime.khash(RD) isa UInt # test that hashing doesn't segfault
    @test AMDGPU.Runtime.khash(RD_view) isa UInt # test that SubArray hashing works
end

@testset "accumulate" begin
  for n in (0, 1, 2, 3, 10, 10_000, 16384, 16384+1) # small, large, odd & even, pow2 and not
    @test testf(x->accumulate(+, x), rand(n))
    @test testf((x,y)->accumulate(+, x; init=y), rand(n), rand())
  end

  # multidimensional
  for (sizes, dims) in ((2,) => 2,
                        (3,4,5) => 2,
                        (1, 70, 50, 20) => 3)
    @test testf(x->accumulate(+, x; dims=dims), rand(Int, sizes))
  end

  # using initializer
  for (sizes, dims) in ((2,) => 2,
                        (3,4,5) => 2,
                        (1, 70, 50, 20) => 3)
    @test testf((x,y)->accumulate(+, x; dims=dims, init=y), rand(Int, sizes), rand(Int))
  end

  # in place
  @test testf(x->(accumulate!(+, x, copy(x)); x), rand(2))

  # specialized
  @test testf(cumsum, rand(2))
  @test testf(cumprod, rand(2))
end

