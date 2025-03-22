    function newsign= ssgn(y) 
           newsign=sign(y);
           Idx=find(abs(y)<1e-12);
           newsign(Idx)=1;

end

