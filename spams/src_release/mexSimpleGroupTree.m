% 
% Usage:   gstruct =mexSimpleGroupTree(degrees)
%
% Name: mexSimpleGroupTree
%
% Description: makes a structure representing a tree given the
%   degree of each level.
%
% Inputs: degrees:  int vector; degrees(i) is the number of children of each node at level i
%
% Output: group_struct: cell array, one element for each node
%                an element is itsel a 4 elements cell array :
%	          nodeid (int >= 0), weight (double), array of vars attached to the node
%                  (here equal to [nodeid]), array of children (nodeid's)
%
% Author: Jean-Paul CHIEZE, 2012
