

function Zeeman!(H_eff::CuArray{Float32, 4}, H_ext::Function, t::Float64)

    H_eff .+= H_ext(t)
    
end

# TODO: Add constant field term 
# TODO: Add spatially dependent term 