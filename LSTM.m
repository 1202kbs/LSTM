% Parameters:
% # of features for each timestep : n (= # of input  units)
% # of training examples          : m
% # of timestep per example       : T
% # of classes                    : c (= # of output units = # of output per timestep)
% 
% For Multi-Class Labeling (n to c)
% X = m X n X T
% z = m X c X T
% 
% X(m,:,T) --> z(m,:,T) correspondence
%
% For Multi-Class Labeling (1 to c)
% X = m X 1 X T
% z = m X c X T
%
% X(m,1,T) --> z(m,:,T) correspondence

momentum = 0.2;
learning_rate = 0.2;

x = linspace(0,4*pi,200);
y = (sin(x) + 1) / 2;

X = zeros(191,1,10);
z = zeros(191,1,10);

for m = 1:191
  for t = 1:10
    X(m,1,t) = x(t + m - 1);
    z(m,1,t) = y(t + m - 1);
  endfor
endfor

I = size(X,2); % Input
H = 1; % # of Memory Cells
K = size(z,2); % Output

w_il = randInitWeights(I,H); % Deal with additional LSTM block by making weights 3D
w_hl = randInitWeights(H,H);
w_cl = randInitWeights(H,H);

w_if = randInitWeights(I,H);
w_hf = randInitWeights(H,H);
w_cf = randInitWeights(H,H);

w_ic = randInitWeights(I,H);
w_hc = randInitWeights(H,H);

w_iw = randInitWeights(I,H);
w_hw = randInitWeights(H,H);
w_cw = randInitWeights(H,H);

w_ck = randInitWeights(H,K);

%Training

for i = 1:1000
  %Forward Propagate
  [a_tl b_tl a_tf b_tf a_tc s_tc a_tw b_tw b_tc a_tk b_tk] = ...
  LSTM_forward(w_il, w_hl, w_cl, w_if, w_hf, w_cf, w_ic, w_hc, w_iw, w_hw, w_cw, w_ck, X);
  %Backpropagate
  [w_il_grad w_hl_grad w_cl_grad w_if_grad w_hf_grad w_cf_grad ... 
  w_ic_grad w_hc_grad w_iw_grad w_hw_grad w_cw_grad w_ck_grad] = ... 
  LSTM_backpropagate(w_cl, w_cf, w_hc, w_cw, w_ck, ... 
  a_tl, b_tl, a_tf, b_tf, a_tc, s_tc, a_tw, b_tw, b_tc, a_tk, b_tk, X, z);
  %Update Weights
  w_ck = momentum * w_ck - learning_rate * w_ck_grad;
  
  w_il = momentum * w_il - learning_rate * w_il_grad;
  w_hl = momentum * w_hl - learning_rate * w_hl_grad;
  w_cl = momentum * w_cl - learning_rate * w_cl_grad;
  
  w_if = momentum * w_if - learning_rate * w_if_grad;
  w_hf = momentum * w_hf - learning_rate * w_hf_grad;
  w_cf = momentum * w_cf - learning_rate * w_cf_grad;
  
  w_ic = momentum * w_ic - learning_rate * w_ic_grad;
  w_hc = momentum * w_hc - learning_rate * w_hc_grad;
  
  w_iw = momentum * w_iw - learning_rate * w_iw_grad;
  w_hw = momentum * w_hw - learning_rate * w_hw_grad;
  w_cw = momentum * w_cw - learning_rate * w_cw_grad;

endfor

[a_test_tl b_test_tl a_test_tf b_test_tf a_test_tc s_test_tc ...
a_test_tw b_test_tw b_test_tc a_test_tk b_test_tk] = ...
LSTM_forward(w_il, w_hl, w_cl, w_if, w_hf, w_cf, w_ic, w_hc, w_iw, w_hw, w_cw, w_ck, test);

plot(y,"color","blue");
hold;
p = zeros(1,200);
p(1:10) = b_test_tk(1,1,:);
for m = 2:191
  p(9 + m) = b_test_tk(m,1,10);
endfor
plot(p,"color","red");
