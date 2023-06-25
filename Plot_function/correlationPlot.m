% correlationPlot Wrapper function for Balnd-Altman to generate only the
% correlation plot. For more infomation please refer to the help on
% BlandAltman.
%
% See also BlandAltman
% By Ran Klein, The Ottawa Hospital, Department of Nuclear Medicine
function [r, fig, stats] = correlationPlot(varargin)
varargin = [varargin, 'ProcessCorrelationOnly','On'];
[~, fig, stats] = BlandAltman(varargin{:});
r = stats.r;
