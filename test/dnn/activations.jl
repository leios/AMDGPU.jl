@testset "Activation functions" begin
    for (T, atol) in ((Float16, 1f-3), (Float32, 1f-6))
        x, dy = randn(T, 16), randn(T, 16)
        xd, dyd = ROCArray(x), ROCArray(dy)

        yd = MIOpen.relu(xd)

        yd = MIOpen.leakyrelu(xd, 0.1)

        yd = MIOpen.softrelu(xd)

        yd = MIOpen.clippedrelu(xd, 6)

        yd = MIOpen.elu(xd, 0.1)

        y = abs.(x)
        yd = MIOpen.abs(xd)
        @test all(isapprox.(Array(yd), y; atol))

        yd = MIOpen.sigmoid(xd)

        y = tanh.(x)
        yd = MIOpen.tanh(xd)
        @test all(isapprox.(Array(yd), y; atol))

        # Non-negative values.

        x, dy = rand(T, 16), rand(T, 16)
        xd, dyd = ROCArray(x), ROCArray(dy)
        α, β, γ = T(1), T(1.1), T(1.2)

        y = (α .+ β .* x).^γ
        yd = MIOpen.power(xd, γ; α, β)
        @test all(isapprox.(Array(yd), y; atol=T == Float16 ? 1f-2 : atol))
    end
end
