% determine in what phase quadrant  to operate
    function quadrant_swap(X,Q)
       M = Q==1 | Q==3;
       X(M) = -imag(X(M)) + 1i*real(X(M));
       X(Q>1) = -X(Q>1);
    end

   