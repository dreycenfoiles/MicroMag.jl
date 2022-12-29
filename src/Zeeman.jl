
function Zeeman!(Heff::CuArray{Float32, 4}, Bext::Vector{Float64}, t::Float64)

    Heff[1, :, :, :] .+= Bext[1] / 1e18 / μ₀
    Heff[2, :, :, :] .+= Bext[2] / 1e18 / μ₀
    Heff[3, :, :, :] .+= Bext[3] / 1e18 / μ₀

    nothing

end


function Zeeman!(Heff::CuArray{Float32,4}, Bext::CuArray{Float32, 4}, t::Float64)
    
    Heff[1, :, :, :] .+= Bext[1, :, :, :] ./ 1e18 ./ μ₀
    Heff[2, :, :, :] .+= Bext[2, :, :, :] ./ 1e18 ./ μ₀
    Heff[3, :, :, :] .+= Bext[3, :, :, :] ./ 1e18 ./ μ₀
    
    nothing

end

function Zeeman!(Heff::CuArray{Float32,4}, Bext::Function, t::Float64)


    Heff .+= Bext(t)[1, :, :, :] ./ 1e18 ./ μ₀
    Heff .+= Bext(t)[2, :, :, :] ./ 1e18 ./ μ₀
    Heff .+= Bext(t)[3, :, :, :] ./ 1e18 ./ μ₀

    nothing

end
