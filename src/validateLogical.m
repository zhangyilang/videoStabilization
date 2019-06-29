function validateLogical(value, varName)
% validation of logical parameter value

%#codegen
%#ok<*EMTC>
%#ok<*EMCA>


if isempty(coder.target)
    % use try/catch to throw error from calling function. This produces an
    % error stack that is better associated with the calling function.
    try 
        localValidate(value, varName)
    catch E        
        throwAsCaller(E); % to produce nice error message from caller.
    end
else
    localValidate(value, varName);
end

function localValidate(value,varName)
validateattributes(value, {'logical','numeric'},...
    {'nonnan', 'scalar', 'real','nonsparse'},...
    mfilename,varName);