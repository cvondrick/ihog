#ifndef __MYGRAPH_H
#define __MYGRAPH_H
#include <stdlib.h>
#include <sys/types.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <linalg.h>
#include <ctype.h>
#include <stdio.h>

/* **************************** */
// functions to build graph and tree struct for fista and proximal
/* the data structure used to build graph or tree is a vector of nodes,
given in any order.
Each node is a StructNodeElem (see below).
In python, it is represented by a list of 4-uples (numbers and numpy vectors)
In R, it is a vector of heterogenous vectors.
In matlab, it is a cell array of cells.
*/
template<typename T> class StructNodeElem {
 public:
 StructNodeElem() :node_num(-1),weight(0.), vars(static_cast<std::vector<int> *>(0)),
    children(static_cast<std::vector<int> *>(0)) {};
 StructNodeElem(int node_num,double weight,std::vector<int> *vars, std::vector<int> *children) :
  node_num(node_num),weight(weight), vars(vars), children(children)  {};
  ~StructNodeElem() {
    if(vars != static_cast<std::vector<int> *>(0)) delete vars;
    if(children != static_cast<std::vector<int> *>(0)) delete children;
  }
  int node_num;
  T weight;
  std::vector<int> *vars;
  std::vector<int> *children;
};

template<typename T>inline T* 
newzeros(int nb) {
  T *p = new T[nb];
  for(int i = 0;i < nb;i++)
    p[i] = static_cast<T>(0);
  return p;
}

inline int* 
m_ones(int nb) {
  int *p = new int[nb];
  for(int i = 0;i < nb;i++) p[i] = -1;
  return p;
}

template<typename T>
inline void del_gstruct(std::vector<StructNodeElem<T> *> *gstruct) {
  if (gstruct == static_cast<std::vector<StructNodeElem<T> *> *>(0)) return;
  for(int i = 0;i < gstruct->size();i++) 
    delete (*gstruct)[i];
  delete gstruct;
}

/* 
   skip spaces in a C string.
   Returns NULL if the string ends with a space
   a pointer to the 1st non-space char otherwise
   
 */
inline char * skip_space(char * s) {
  char *ps = s;
  while(isspace(*ps))
    ps++;
  if(*ps == 0)
    return (char *)0;
  else
    return ps;
}

/* parse a line describing a node
 id weight [ list-of-vars ] -> list-of-children
 weight and list-of-children may be omitted; list-of-vars may be empty
 Input :
     s : C string to parse
     vresult : result vector of 4 std::string (one per line component).
          an empty element hans an empty string
 Return:
    -1 if error, 0 if OK
 */
static int parse_line(char *s,std::vector<std::string> &vresult) {
  char *p = skip_space(s);
  if(p == NULL) return -1;
  for(int i = 0;i < 4;i++) vresult.push_back(std::string(""));
  // node id
  char *p0 = p;
  int n = 0;
  while(! isspace(*p) && (*p != 0)) {
    if(! isdigit(*p)) return -1;
    p++;
    n++;
  }
  vresult[0] = std::string(p0,n);
  // weight
  if((p = skip_space(p)) == NULL) return -1;
  if(*p != '[') {
    p0 = p;
    n = 0;
    while(! isspace(*p)) {
      if(!isprint(*p)) return -1;
      p++;
      n++;
    }
    vresult[1] = std::string(p0,n);
  }
  // variables list
  if((p = skip_space(p)) == NULL) return -1;
  if(*p != '[') return -1;
  p++;
  if((p = skip_space(p)) == NULL) return -1;
  p0 = p;
  n = 0;
  while(*p != ']') {
    if(*p == 0) return -1;
    p++;
    n++;
  }
  if (n > 0) {
    char *p1 = p - 1;
    while(isspace(*p1)) {
      p1--;
      n--;
    }
    vresult[2] = std::string(p0,n);
  }
  p++;
  // children list
  if((p = skip_space(p)) == NULL) return 0;
  if(*p != '-' || *(p+1) != '>') return -1;
  p += 2;
  if((p = skip_space(p)) == NULL) return 0;
  // children list is not empty
  p0 = p;
  n = 0;
  while (*p != 0) {
    p++;
    n++;
  }
  p--;
  while(isspace(*p)) {
    p--;
    n--;
  }
  vresult[3] = std::string(p0,n);
  return 0;
}

// given a string of int's separated by spaces, returns a vector of values
inline std::vector<int> *
intlist(string s) {
  std::vector<int> *result = new std::vector<int>;
  // first remove trailing spaces
  int n = s.size() - 1;
  while (n >= 0) {
    if (s[n] != ' ') break;
    n--;
  }
  if(n < 0) return result;
  s.resize(n + 1);

  std::istringstream is(s);
  int i;
  while(1) {
    is >> i;
    result->push_back(i);
    if(is.eof()) break;
  }
  return result;
}
/*
  Reads a text file describing "simply" the structure of groups
  of variables needed by mexProximalGraph, mexProximalTree, mexFistaGraph,
   mexFistaTree and mexStructTrainDL
   Ecah line describes a group of variables as a node of a tree.
      Each line has up to 4 fields separated by spaces:
       node-id node-weight [variables-list] -> children-list
    Let's define Ng = number of groups, and Nv = number of variables.
    node-id must be in the range (0 - Ng-1), and there must be Ng nodes
    weight is a float (1. if omitted)
    variables-list : a space separated list of integers, maybe empty,
        but '[' and '] must be present. Numbers in the range (0 - Nv-1)
    children-list : a space separated list of node-id's
             If the list is empty, '->' may be omitted.
  The data must obey some rules : 
  A group contains the variables of the corresponding node and of the whole subtree.
  Variables attached to a node are those that are not int the subtree.
 If the data destination is a Graph, there may be several independant trees,
     and a varibale may appear in several trees.
 If the destination is a Tree, there must be only one tree, the root node
   must have id == 0 and each variable must appear only once.
 Args: filename
 Return : a vector of StructNodeElem<T>
*/
template <typename T>
std::vector<StructNodeElem<T> *> *_groupStructOfString(const char *data) throw(const char *){
  std::istringstream is(data);
  std::vector<StructNodeElem<T> *> *gstruct = new std::vector<StructNodeElem<T> *>;

  int ret;
  char buffer[1024];
  while(! is.eof()) {
    is.getline(buffer,1024);
    //  while(fgets(buffer,1024,f) != NULL) {
    if(buffer[0] == '#' || buffer[0] == 0) continue;
    std::vector<string> lst;
    if(parse_line(buffer,lst) < 0) {
      char tmp[1024];
      snprintf(tmp,1024,"Bad inode line <%s>\n",buffer);
      throw(tmp);
    }
    int inode = atoi(lst[0].c_str());
    T weight = (lst[1].size() > 0) ? static_cast<T>(atof(lst[1].c_str())) : 1.;
    std::vector<int> *vars = intlist(lst[2]);
    std::vector<int> *children = intlist(lst[3]);
    StructNodeElem<T> *node = new StructNodeElem<T>(inode,weight,vars,children);
    gstruct->push_back(node);
  }
  return gstruct;
}
template <typename T>
std::vector<StructNodeElem<T> *> *_readGroupStruct(const char *file) throw(const char *){
  std::ifstream infile;
  infile.open (file, ifstream::in);
  if(! infile.good())
    throw("readGroupStruct: cannot open file");
  infile.seekg (0, ios::end);
  int length = infile.tellg();
  infile.seekg (0, ios::beg);
  char *buffer = new char[length +1];
  infile.read(buffer,length);
  infile.close();
  buffer[length] = 0;
  std::vector<StructNodeElem<T> *> *gstruct = _groupStructOfString<T>(buffer);
  delete[] buffer;
  return gstruct;
}

/*
 Make StructNodeElem vector representing a tree given the
   degree of each level.

 Args: degrees: an int pointer to the number of children at each level
         n : nb of levles
 Return: StructNodeElem vector
 */
template <typename T>
std::vector<StructNodeElem<T> *> *_simpleGroupTree(int *degr, int n) throw(const char *){
  std::vector<int> degrees;
  for(int i = 0;i < n;i++)
    degrees.push_back(degr[i]);
  degrees.push_back(0);
  std::vector<StructNodeElem<T> *> *gstruct = new std::vector<StructNodeElem<T> *>;
  int nb_levels = degrees.size();
  if(nb_levels <= 1)
    throw("simpleGroupTree: number of levels must be > 1");
  // index of 1st node at each level
  int *level_starts = newzeros<int>(nb_levels);
  int nb(1), k(1);
  for (int i = 1;i < nb_levels;i++) {
    nb *= degrees[i - 1];
    level_starts[i] = k;
    k += nb;
  }
  int nb_nodes = 1;
  for(int i = nb_levels - 2;i >= 0;i--) {
    nb_nodes = nb_nodes * degrees[i] + 1;
  }
  //  cout << "LEVELS " << nb_levels << " NODES " << nb_nodes << endl;
  std::vector<int> lst(1,0);
  k = 0;
  for(int id = 0;id < nb_levels;id++) {
    if ((id + 1) < nb_levels)
      k = level_starts[id + 1];
    std::vector<int> lstx;
    for(std::vector<int>::iterator it = lst.begin();it < lst.end();it++) {
      int inode = *it;
      std::vector<int> *children = new std::vector<int>;
      for (int j = 0;j < degrees[id];j++) {
	children->push_back(k);
	lstx.push_back(k);
	k++;
      }
      StructNodeElem<T> *node = new StructNodeElem<T>(inode,1.,new std::vector<int>(1,inode),children);
      gstruct->push_back(node);
    }
    lst = lstx;
  }
  delete []level_starts;
  return gstruct;
}

/* 
   verifies the validity of a group structure
   Args: gstruct : the group structure as a StructNodeElem<T> vector
          tree_mode : if true, the data must satisfy Tree constraints
	  nbVars : pointer to a location where to put the nb of variables.
   Return: true if data is OK

 */
template <typename T>
bool checkGroupTree(std::vector<StructNodeElem<T> *> *gstruct, bool tree_mode, int *nbVars) {

  if (gstruct == static_cast<std::vector<StructNodeElem<T> *> *>(0)) {
    *nbVars = 0;
    return false;
  }
  int nbgr = gstruct->size();
  int nb_vars = 0;
  bool is_ok = true;
  // table to check presence/duplication  of groups
  int *groups = newzeros<int>(nbgr);
  int *used_children = newzeros<int>(nbgr); // to check unicity of children
  std::vector<int> all_vars;
  int inode = 0;
  bool ordered = true;
  for(typename std::vector<StructNodeElem<T> *>::iterator it = gstruct->begin();it != gstruct->end();it++) {
    StructNodeElem<T> *node = *it;
    T weight = node->weight;
    int i = node->node_num;
    if (weight < 0.) {
      std::cout << "Incorrect weight (" << weight << ") for group " << inode << std::endl;
      is_ok = false;
      continue;
    }
    if (i != inode)
      ordered = false;  // gstruct is not sorted by inode
    if (i < 0 || i > nbgr) {
      std::cout << "Bad group num " << i << " (should be in [0 - " << nbgr << "\n";
      is_ok = false;
      break;
    }
    inode++;
    if(groups[i] != 0) {
      std::cout << "BAD GSTRUCT : duplicate group " << i << std::endl;
      is_ok = false;
      break;
    }
    groups[i] = 1;
    std::vector<int> *vars = node->vars;
    std::vector<int> *children = node->children;
    if (vars->size() == 0 && children->size() == 0) {
      std::cout << "BAD node " << i << " : has neither vars nor children\n";
      continue;
    }
    all_vars.insert(all_vars.end(),vars->begin(),vars->end());
    for(std::vector<int>::iterator it = children->begin();it != children->end();it++) {
      if(tree_mode && *it == 0) {
	std::cout << "group 0 must be the root of the tree\n";
	is_ok = false;
	break;
      }
      if (*it < 0 || *it >= nbgr || used_children[*it] != 0) {
	std::cout << "Bad child " << *it << " for " << i << std::endl;
	is_ok = false;
	break;
      }
      used_children[*it] = 1;
    }
  }
  delete[] groups;
  delete[] used_children;

  if (is_ok) {
    // check variables
    // for graph struct, unicity of subtrees is checked when building groups and group_vars
    int max_var = 0;
    for(int i = 0;i < all_vars.size();i++) 
      if (all_vars[i] > max_var) max_var = all_vars[i];
    nb_vars = max_var + 1;
    if(nb_vars > all_vars.size()) {
      std::cout << "There are missing variables\n";
      *nbVars = all_vars.size();
      return false;
    }
    int *used_vars = newzeros<int>(nb_vars);
    for(int i = 0;i < all_vars.size();i++) {
      int v = all_vars[i];
      if(v < 0 || v >= nb_vars) {
	std::cout << "Bad var " << v << "(not in [0-" << nb_vars << "]\n";
	is_ok = false;
	continue;
      }
      if(tree_mode && used_vars[v] != 0) {
	std::cout << "Duplicate var " << v << std::endl;
	is_ok = false;
	continue;
      }
      used_vars[v] = 1;
    }
    for(int i = 0;i < nb_vars;i++)
      if (used_vars[i] == 0) 
	std::cout << "Missing var " << i << std::endl;
    delete[] used_vars;
  }
  *nbVars = nb_vars;
  return is_ok;
}
template <typename T>
static void loop_tree(std::vector<int> *lst,int *new_i, int *nb_vars,
		      std::vector<StructNodeElem<T> *> *gstruct,int *group_index,int *nodes_index) {
  if(lst->size() == 0) return;
  for(std::vector<int>::iterator it = lst->begin();it != lst->end();it++) {
    int igroup = *it;
    group_index[igroup] = *new_i;
    *new_i += 1;
    StructNodeElem<T> *node = (*gstruct)[nodes_index[igroup]];
    *nb_vars += node->vars->size();
    if (node->children->size() != 0)
      loop_tree<T>(node->children,new_i,nb_vars,gstruct,group_index,nodes_index);
  }
}

/*
  The tree structure used by proximalTree fistaTree or structTrainDL suppose a particular
  order in the variables.
This function will put correct values for each group ang give the inverse permutaion
to apply to the result of these programs
 */
template <typename T>
static std::vector<StructNodeElem<T> *> *reorder_group_tree(std::vector<StructNodeElem<T> *> *gstruct,int **pvar_inv) {
  int nb_nodes = gstruct->size();
  int *nodes_index = newzeros<int>(nb_nodes);
  int i = 0;
  for(typename std::vector<StructNodeElem<T> *>::iterator it = gstruct->begin();it != gstruct->end();it++,i++) {
    StructNodeElem<T> *node = *it;
    nodes_index[node->node_num] = i;
  }
  int *group_index = m_ones(nb_nodes);
  int nb_vars = 0, new_i = 0;
  std::vector<int> lst(1,0);
  loop_tree<T>(&lst,&new_i,&nb_vars,gstruct,group_index,nodes_index);
  int *group_inv = newzeros<int>(nb_nodes);
  for(int i = 0;i < nb_nodes;i++) 
    group_inv[group_index[i]] = i;
  int *vars_inv = m_ones(nb_vars);
  *pvar_inv = vars_inv;
  int ivar = 0;
  std::vector<StructNodeElem<T> *> *newgstruct = new std::vector<StructNodeElem<T> *>;
  for (int j=0;j < nb_nodes;j++) {
    i = group_inv[j];
    StructNodeElem<T> *node = (*gstruct)[i];
    std::vector<int> *nv = new std::vector<int>;
    int n = node->vars->size();
    if(n > 0) {
      std::vector<int> *vars = node->vars;
      for(int k = 0;k < n;k++) {
	vars_inv[ivar] = (*vars)[k];
	nv->push_back(ivar);
	ivar++;
      }
    }
    std::vector<int> *nchild = new std::vector<int>;
    for(typename std::vector<int>::iterator it = node->children->begin();it != node->children->end();it++)
      nchild->push_back(group_index[*it]);
    StructNodeElem<T> *node2 = new StructNodeElem<T>(group_index[node->node_num],node->weight,nv,nchild);
    newgstruct->push_back(node2);
  }
  delete[] nodes_index;
  delete[] group_index;
  delete[] group_inv;
  return newgstruct;
}

/* 
   Input args : 
     gstruct = vector of StructNodeElem<T> representing the groups structure
        as a bunch of trees
        (Each element describes a node of a tree)
   Output args:
	pgroups : matrix of groups inclusions
	       groups(j,i) : true if group j is included in group i (gj in subtree of gi)
	groups_var : matrix of variables
	       groups_var(j,i) : true if var j is in group i
   Return : weights associated to groups (eta_g)
       NB: eta_g groups own_variables N_own_variables are described in detail in
       the documantation of proximalTree.
 */
template <typename T>
Vector<T> *_graphOfGroupStruct(std::vector<StructNodeElem<T> *> *gstruct,SpMatrix<bool> **pgroups,SpMatrix<bool> **pgroups_var) throw(const char *) {
  int nb_vars;
  Vector<T> *peta_g;
  if (! checkGroupTree<T>(gstruct,false,&nb_vars))
    throw("graphOfGroupStruct: bad input data");
  int nb_groups = gstruct->size();
  bool *dgroups = newzeros<bool>(nb_groups * nb_groups);
  bool *dgroups_var = newzeros<bool>(nb_vars * nb_groups);
  T *deta_g = new T[nb_groups];
  // NB : matrix are stored by columns
  for(typename std::vector<StructNodeElem<T> *>::iterator it = gstruct->begin();it != gstruct->end();it++) {
    StructNodeElem<T> *node = *it;
    int i = node->node_num;
    deta_g[i] = node->weight;
    std::vector<int> *vars = node->vars;
    std::vector<int> *children = node->children;
    bool *pv = dgroups_var + (nb_vars * i);
    for(int j = 0;j < vars->size();j++) {
      int k = (*vars)[j];
      *(pv + k) = 1;
    }
    bool *pg = dgroups + (nb_groups * i);
    for(int k = 0;k < children->size();k++) {
      int j = (*children)[k];
      *(pg + j) = 1;
    }
  }
  // now we need to verify that variabales appear only in one tree
  bool is_ok = true;
  for(int ivar = 0;ivar < nb_vars;ivar++) {
    std::vector<int> l;
    bool *pv = dgroups_var + ivar;
    for(int g = 0;g < nb_groups;g++)
      if( *(pv + (g * nb_vars)))
	l.push_back(g);
    if(l.size() > 1) {
      for(int k = 0;k < l.size();k++) {
	int g = l[k];
	bool *pg = dgroups + g;
	for(int j = 0;j < nb_groups;j++) {
	  if( *(pg+ (j * nb_groups)))   // g included in j
	    if(*(pv + j) != 0) {
	      std::cout << "Duplicate var " << ivar << " in " << g << " and " << j << std::endl;
	      is_ok = false;
	    }
	}
      }
    }
  }
  if(! is_ok) throw("graphOfGroupStruct: bad input data");
  peta_g = new Vector<T>(nb_groups);
  memcpy(peta_g->rawX(),deta_g,nb_groups * sizeof(T));
  delete []deta_g;
  Matrix<bool> *tmp = new Matrix<bool>(dgroups_var,nb_vars,nb_groups);
  SpMatrix<bool> *groups_var = new SpMatrix<bool>();
  tmp->toSparse((SpMatrix<bool> &)(*groups_var));
  delete tmp;
  delete[] dgroups_var;
  tmp = new Matrix<bool>(dgroups,nb_groups,nb_groups);
  SpMatrix<bool> *groups = new SpMatrix<bool>();
  tmp->toSparse((SpMatrix<bool> &)(*groups));
  delete tmp;
  delete[] dgroups;
  *pgroups_var = groups_var;
  *pgroups = groups;
  return peta_g;
}

/* 
   Input args : 
     gstruct = vector of StructNodeElem<T> representing the groups structure
        as a bunch of trees
        (Each element describes a node of a tree)
   Output args:
        pperm : NULL or address of the inverse permutation table of variables
	pnb_vars : 0 if pperm is NULL, else nb of variables 
	eta_g : vector of weights
	pgroups : matrix of groups inclusions
	pown_variables, pN_own_variables : tables describing the distribution
	   of variables in groups
       NB: eta_g groups own_variables N_own_variables are described in detail in
       the documantation of proximalTree.
   Return : nb of variables
 */
template <typename T>
int _treeOfGroupStruct(std::vector<StructNodeElem<T> *> *gstruct,int **pperm,int *pnb_vars,Vector<T> **peta_g,SpMatrix<bool> **pgroups,Vector<int> **pown_variables,Vector<int> **pN_own_variables) throw(const char *){
  int nb_vars;
  *pnb_vars = 0;
  if (! checkGroupTree<T>(gstruct,true,&nb_vars))
    throw("treeOfGroupStruct: bad input data");
  int nb_groups = gstruct->size();
  bool *dgroups = newzeros<bool>(nb_groups * nb_groups);
  T *deta_g = new T[nb_groups];
  int *down_variables = newzeros<int>(nb_groups);
  int *dN_own_variables = newzeros<int>(nb_groups);
  int *var_inv;
  std::vector<StructNodeElem<T> *> *ngstruct = reorder_group_tree<T>(gstruct,&var_inv);
  for(typename std::vector<StructNodeElem<T> *>::iterator it = gstruct->begin();it != gstruct->end();it++) {
    StructNodeElem<T> *node = *it;
    int i = node->node_num;
    deta_g[i] = node->weight;
    std::vector<int> *vars = node->vars;
    std::vector<int> *children = node->children;
    int n = vars->size();
    dN_own_variables[i] = n;
    if (n > 0)
      down_variables[i] = (*vars)[0];
    bool *pg = dgroups + (i * nb_groups);
    for(int j = 0; j < children->size();j++) {
      int k = (*children)[j];
      *(pg + k) = true;
    }
  }
  // set own_variables when N_own_variables is 0
  for(int i = nb_groups - 2;i >= 0;i--)
    if(dN_own_variables[i] == 0)
      down_variables[i] = down_variables[i+1];
  // 
  int *perm = static_cast<int *>(0);
  int i = 0;
  for(int k = 0;k < nb_vars;k++,i++) {
    int j = var_inv[k];
    if(j != i) {
      perm = var_inv;
      break;
    }
  }
  if(perm == static_cast<int *>(0))
    delete[] var_inv;
  else
      *pnb_vars = nb_vars;
  *pperm = perm;
  *peta_g = new Vector<T>(nb_groups);
  memcpy((*peta_g)->rawX(),deta_g,nb_groups * sizeof(T));
  delete []deta_g;
  Matrix<bool> *tmp = new Matrix<bool>(dgroups,nb_groups,nb_groups);
  SpMatrix<bool> *groups = new SpMatrix<bool>();
  tmp->toSparse((SpMatrix<bool> &)(*groups));
  delete tmp;
  delete[] dgroups;
  *pgroups = groups;
  *pown_variables = new Vector<int>(nb_groups);
  memcpy((*pown_variables)->rawX(),down_variables,nb_groups * sizeof(int));
  delete [] down_variables;
  *pN_own_variables = new Vector<int>(nb_groups);
  memcpy((*pN_own_variables)->rawX(),dN_own_variables,nb_groups * sizeof(int));
  delete [] dN_own_variables;
  del_gstruct(ngstruct);
  return nb_vars;
}

#endif /* __MYGRAPH_H */
