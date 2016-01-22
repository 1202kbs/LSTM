function W = randInitWeights(L_in, L_out)

W = zeros(L_in, L_out);

W = rand(L_in, L_out) * 2 - 1;

end
