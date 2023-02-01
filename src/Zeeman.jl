struct Zeeman{T} <: AbstractField
    B_ext::T
end

function (zeeman::Zeeman{T})(H_eff::CuArray{Float32}, m::CuArray{Float32}, t) where T<:Vector

    H_eff[1, ..] .+= zeeman.B_ext[1] / 1e18 / μ₀
    H_eff[2, ..] .+= zeeman.B_ext[2] / 1e18 / μ₀
    H_eff[3, ..] .+= zeeman.B_ext[3] / 1e18 / μ₀

    nothing
end


function (zeeman::Zeeman{T})(H_eff::CuArray{Float32}, m::CuArray{Float32}, t) where T<:Function

    Bx, By, Bz = zeeman.B_ext(t)

    H_eff[1, ..] .+= Bx / 1e18 / μ₀
    H_eff[2, ..] .+= By / 1e18 / μ₀
    H_eff[3, ..] .+= Bz / 1e18 / μ₀

    nothing

end