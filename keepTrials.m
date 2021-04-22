function keep = keepTrials(rejTrials,nonRejTrials)
    keep = [false,false,false,false];
    nonRejectStart=1;
    rejIndex=1;
    
    % Stop when all items in the rejected array considered
    while (rejIndex <= size(rejTrials,2))
        nonRejIndex = nonRejectStart;
        while (nonRejIndex <= size(nonRejTrials,2))
            % compare trials 
            keep(nonRejIndex) = isequal(rejTrials(:,rejIndex),nonRejTrials(:,nonRejIndex));
            
            % if  they not the same, trial is rejected move onto the next
            % element in the non-rejected array
            if(~keep(nonRejIndex))
                nonRejIndex = nonRejIndex + 1;
            else
                % If they are the same vectors, this is not rejected,
                % move on to the next element in the rejected array
                % and start comparing from the next element that is not
                % compared (since this is the first time it lands here, all
                % others must be rejected
                nonRejectStart = nonRejIndex+1;
                rejIndex = rejIndex +1;
                
                % Exit this loop if found a non-rejected trial
                break
            end
        end
    end    
end