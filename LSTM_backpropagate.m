function [w_il_grad w_hl_grad w_cl_grad w_if_grad w_hf_grad w_cf_grad ... 
w_ic_grad w_hc_grad w_iw_grad w_hw_grad w_cw_grad w_ck_grad] = ... 
LSTM_backpropagate(w_cl, w_cf, w_hc, w_cw, w_ck, ... 
a_tl, b_tl, a_tf, b_tf, a_tc, s_tc, a_tw, b_tw, b_tc, a_tk, b_tk, X, z)

I = size(X, 2);
H = size(w_ck, 1);
K = size(w_ck, 2);

T = size(X, 3);
m = size(X, 1);

delta_tk = zeros(m,K,T);
e_tc     = zeros(m,H,T);
delta_tw = zeros(m,H,T);
e_ts     = zeros(m,H,T);
delta_tc = zeros(m,H,T);
delta_tf = zeros(m,H,T);
delta_tl = zeros(m,H,T);

w_il_grad = zeros(I,H);
w_hl_grad = zeros(H,H);
w_cl_grad = zeros(H,H);

w_if_grad = zeros(I,H);
w_hf_grad = zeros(H,H);
w_cf_grad = zeros(H,H);

w_ic_grad = zeros(I,H);
w_hc_grad = zeros(H,H);

w_iw_grad = zeros(I,H);
w_hw_grad = zeros(H,H);
w_cw_grad = zeros(H,H);

w_ck_grad = zeros(H,K);

for t = T:-1:1
  if (t == T)
    delta_tk(:,:,t) = b_tk(:,:,t) - z(:,:,t);
    
    e_tc(:,:,t) = delta_tk(:,:,t) * w_ck';
    
    delta_tw(:,:,t) = sigmoidGradient(a_tw(:,:,t)) .* (sum(sigmoid(s_tc(:,:,t)) .* e_tc(:,:,t), 2));
    
    e_ts(:,:,t) = b_tw(:,:,t) .* sigmoidGradient(s_tc(:,:,t)) .* e_tc(:,:,t) + delta_tw(:,:,t) * w_cw';
    
    delta_tc(:,:,t) = b_tl(:,:,t) .* sigmoidGradient(a_tc(:,:,t)) .* e_ts(:,:,t);
    
    delta_tf(:,:,t) = sigmoidGradient(a_tf(:,:,t)) .* (sum(s_tc(:,:,t-1) .* e_ts(:,:,t), 2));
    
    delta_tl(:,:,t) = sigmoidGradient(a_tl(:,:,t)) .* (sum(sigmoid(a_tc(:,:,t)) .* e_ts(:,:,t), 2));
  else 
    delta_tk(:,:,t) = b_tk(:,:,t) - z(:,:,t);
    
    e_tc(:,:,t) = delta_tk(:,:,t) * w_ck' + delta_tl(:,:,t) * w_cl' + ...
    delta_tf(:,:,t) * w_cf' + delta_tc(:,:,t) * w_hc' + delta_tw(:,:,t) * w_cw';
    
    delta_tw(:,:,t) = sigmoidGradient(a_tw(:,:,t)) .* (sum(sigmoid(s_tc(:,:,t)) .* e_tc(:,:,t), 2));
    
    e_ts(:,:,t) = b_tw(:,:,t) .* sigmoidGradient(s_tc(:,:,t)) .* e_tc(:,:,t) + ...
    b_tf(:,:,t+1) .* e_ts(:,:,t+1) + delta_tl(:,:,t + 1) * w_cl' + ...
    delta_tf(:,:,t+1) * w_cf' + delta_tw(:,:,t) * w_cw';
    
    delta_tc(:,:,t) = b_tl(:,:,t) .* sigmoidGradient(a_tc(:,:,t)) .* e_ts(:,:,t);
    
    if (t == 1)
      delta_tf(:,:,t) = 0;
    else
      delta_tf(:,:,t) = sigmoidGradient(a_tf(:,:,t)) .* (sum(s_tc(:,:,t-1) .* e_ts(:,:,t), 2));
    endif
    
    delta_tl(:,:,t) = sigmoidGradient(a_tl(:,:,t)) .* (sum(sigmoid(a_tc(:,:,t)) .* e_ts(:,:,t), 2));
  endif
endfor

for t = 1:T
  if (t == 1)
    w_il_grad += X(:,:,t)' * delta_tl(:,:,t);
    w_hl_grad += 0;
    w_cl_grad += 0;
    
    w_if_grad += X(:,:,t)' * delta_tf(:,:,t);
    w_hf_grad += 0;
    w_cf_grad += 0;
    
    w_ic_grad += X(:,:,t)' * delta_tc(:,:,t);
    w_hc_grad += 0;
    
    w_iw_grad += X(:,:,t)' * delta_tw(:,:,t);
    w_hw_grad += 0;
    w_cw_grad += s_tc(:,:,t)' * delta_tw(:,:,t);
    
    w_ck_grad += b_tc(:,:,t)' * delta_tk(:,:,t);
  else
    w_il_grad += X(:,:,t)' * delta_tl(:,:,t);
    w_hl_grad += b_tc(:,:,t-1)' * delta_tl(:,:,t);
    w_cl_grad += s_tc(:,:,t-1)' * delta_tl(:,:,t);
    
    w_if_grad += X(:,:,t)' * delta_tf(:,:,t);
    w_hf_grad += b_tc(:,:,t-1)' * delta_tf(:,:,t);
    w_cf_grad += s_tc(:,:,t-1)' * delta_tf(:,:,t);
    
    w_ic_grad += X(:,:,t)' * delta_tc(:,:,t);
    w_hc_grad += b_tc(:,:,t-1)' * delta_tc(:,:,t);
    
    w_iw_grad += X(:,:,t)' * delta_tw(:,:,t);
    w_hw_grad += b_tc(:,:,t-1)' * delta_tw(:,:,t);
    w_cw_grad += s_tc(:,:,t)' * delta_tw(:,:,t);
    
    w_ck_grad += b_tc(:,:,t)' * delta_tk(:,:,t);
  endif
endfor

end
