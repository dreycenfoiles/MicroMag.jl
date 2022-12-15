
function Zeeman!(H_eff::CuArray{Float32,4}, B_ext::CuArray{Float32,1})
    
    H_eff[1, :, :, :] .+= B_ext[1] / 1e18 / μ₀
    H_eff[2, :, :, :] .+= B_ext[2] / 1e18 / μ₀
    H_eff[3, :, :, :] .+= B_ext[3] / 1e18 / μ₀
    
    nothing

end

function Zeeman!(H_eff::CuArray{Float32,4}, B_ext::CuArray{Float32,4})
    
    H_eff .+= B_ext ./ 1e18 ./ μ₀
    
    nothing

end 