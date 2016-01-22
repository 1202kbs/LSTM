function [a_tl b_tl a_tf b_tf a_tc s_tc a_tw b_tw b_tc a_tk b_tk] = ...
LSTM_forward(w_il, w_hl, w_cl, w_if, w_hf, w_cf, w_ic, w_hc, w_iw, w_hw, w_cw, w_ck, X)

I = size(X, 2);
H = size(w_ck, 1);
K = size(w_ck, 2);

T = size(X, 3);
m = size(X, 1);

a_tl = zeros(m,H,T);
b_tl = zeros(m,H,T);

a_tf = zeros(m,H,T);
b_tf = zeros(m,H,T);

a_tc = zeros(m,H,T);
s_tc = zeros(m,H,T);

a_tw = zeros(m,H,T);
b_tw = zeros(m,H,T);

b_tc = zeros(m,H,T);

a_tk = zeros(m,K,T);
b_tk = zeros(m,K,T);

for t = 1:T
  if (t == 1)
    a_tl(:,:,t) = X(:,:,t) * w_il;
    b_tl(:,:,t) = sigmoid(a_tl(:,:,t));
    
    a_tf(:,:,t) = X(:,:,t) * w_if;
    b_tf(:,:,t) = sigmoid(a_tf(:,:,t));
    
    a_tc(:,:,t) = X(:,:,t) * w_ic;
    s_tc(:,:,t) = b_tl(:,:,t) .* sigmoid(a_tc(:,:,t));
    
    a_tw(:,:,t) = X(:,:,t) * w_iw + s_tc(:,:,t) * w_cw;
    b_tw(:,:,t) = sigmoid(a_tw(:,:,t));
    
    b_tc(:,:,t) = b_tw(:,:,t) .* sigmoid(s_tc(:,:,t));
    
    a_tk(:,:,t) = b_tc(:,:,t) * w_ck;
    b_tk(:,:,t) = sigmoid(a_tk(:,:,t));
  else
    a_tl(:,:,t) = X(:,:,t) * w_il + b_tc(:,:,t-1) * w_hl + s_tc(:,:,t-1) * w_cl;
    b_tl(:,:,t) = sigmoid(a_tl(:,:,t));
    
    a_tf(:,:,t) = X(:,:,t) * w_if + b_tc(:,:,t-1) * w_hf + s_tc(:,:,t-1) * w_cf;
    b_tf(:,:,t) = sigmoid(a_tf(:,:,t));
    
    a_tc(:,:,t) = X(:,:,t) * w_ic + b_tc(:,:,t-1) * w_hc;
    s_tc(:,:,t) = b_tf(:,:,t) .* s_tc(:,:,t-1) + b_tl(:,:,t) .* sigmoid(a_tc(:,:,t));
    
    a_tw(:,:,t) = X(:,:,t) * w_iw + b_tc(:,:,t-1) * w_hw + s_tc(:,:,t) * w_cw;
    b_tw(:,:,t) = sigmoid(a_tw(:,:,t));
    
    b_tc(:,:,t) = b_tw(:,:,t) .* sigmoid(s_tc(:,:,t));
    
    a_tk(:,:,t) = b_tc(:,:,t) * w_ck;
    b_tk(:,:,t) = sigmoid(a_tk(:,:,t));
  endif
endfor

end
