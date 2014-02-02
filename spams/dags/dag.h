/*!
/* Software SPAMS v2.3 - Copyright 2009-2011 Julien Mairal 
 *
 * This file is part of SPAMS.
 *
 * SPAMS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SPAMS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SPAMS.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef DAG_H
#define DAG_H

#include <linalg.h>
#include <list.h>

template <typename T>
int count_cc_graph(const SpMatrix<T>& G, Vector<T>& active) {
   INTM nzmax=0;
   for (INTM i = 0; i<active.n(); ++i)
      if (active[i]) nzmax++;
   list_int** cc = new list_int*[nzmax];
   int* pr_list = new int[active.n()];
   memset(pr_list,-1,active.n()*sizeof(int));
   int count=0;
   list_int list;
   for (int i = 0; i<active.n(); ++i) {
      if (active[i]) {
         list.push_back(i);
         pr_list[i]=count;
         cc[count] = new list_int();
         cc[count++]->push_back(i);
      }
   }
   const INTM* pB = G.pB();
   const INTM* r = G.r();
   const T* v = G.v();

   while (!list.empty()) {
      int node=list.front();
      list.pop_front();
      for (int j = pB[node]; j<pB[node+1]; ++j) {
         int child=r[j];
         if (active[child]) {
            if (pr_list[node] != pr_list[child]) {
               // fusion
               const int pr = pr_list[child];
               for (const_iterator_int it = cc[pr]->begin(); it != cc[pr]->end(); ++it) {
                  pr_list[*it]=pr_list[node];
               }
               cc[pr_list[node]]->fusion(*(cc[pr]));
               delete(cc[pr]);
               cc[pr]=NULL;
            }
         }
      }
   }

   int num_cc=0;
   for (int i = 0; i<nzmax; ++i) if (cc[i]) num_cc++;
   for (int i = 0; i<nzmax; ++i) delete(cc[i]);
   delete[](cc);
   delete[](pr_list);
   return num_cc;
}

template <typename T>
void remove_cycles(const SpMatrix<T>& G1, SpMatrix<T>& G2) {
   G2.copy(G1);
   int n = G1.n();
   int* color = new int[n];
   memset(color,0,n*sizeof(int));
   int next=0;
   list_int list;
   INTM* pB = G2.pB();
   INTM* r = G2.r();
   T* v = G2.v();

   list_int current_path;
   bool cycle_detected=true;
   int cycles_removed=0;
   while (cycle_detected) {
      cycle_detected=false;
      list.clear();
      for (int i = 0; i<n; ++i) if (color[i] != 2) {
         color[i]=0;
         list.push_back(i);
      }
      current_path.clear();
      while (!list.empty()) {
         const int node=list.front();
         if (color[node] == 0) {
            current_path.push_front(node);
            color[node]=1; 
            for (int i = pB[node]; i<pB[node+1]; ++i) {
               if (v[i]) {
                  const int child = r[i];
                  if (color[child]==1) {
                     cycle_detected=true;
                     list_int reverse_path;
                     current_path.reverse(reverse_path);
                     while (true) {
                        const int current_node=reverse_path.front();
                        if (current_node == child) break;
                        reverse_path.pop_front();
                     }
                     // remove beginning of the path

                     reverse_path.push_back(child);
                     T min_link = INFINITY;
                     int min_node= -1;
                     const_iterator_int it = reverse_path.begin();
                     // detect weakest link
                     while (true) {
                        const int current_node=*it;
                        ++it;
                        if (it == reverse_path.end()) break;
                        for (int j = pB[current_node]; j<pB[current_node+1]; ++j) {
                           if (r[j]==*it) {
                              if (min_link  > v[j]) {
                                 min_link=v[j];
                                 min_node=j;
                              }
                              break;
                           }
                        }
                     }
                     v[min_node]=0;
                     list.clear();
                     cycles_removed++;
                     cerr << "                    \r" << cycles_removed;
                     break;
                  } else if (color[child]==0) {
                     list.push_front(child);
                  }
               }
            }
         } else if (color[node] == 1) {
            /// means descendants(node) is acyclic
            color[node]=2;
            list.pop_front();
            current_path.pop_front();
         } else if (color[node] == 2) {
            list.pop_front();
         }
      }
   }
   int current_remove=0;
   for (int i = 0; i<n; ++i) {
      int old_remove=current_remove;
      for (int j = pB[i]; j < pB[i+1]; ++j) {
         if (v[j]) {
            r[j-current_remove]=r[j];
            v[j-current_remove]=v[j];
         } else {
            current_remove++;
         }
      }
      pB[i]-=old_remove;
   }
   pB[n]-=current_remove;

   delete[](color);
}

template <typename T>
T count_paths_dags(const SpMatrix<T>& G) {
   const INTM n = G.n();
   T* num_paths = new T[n];
   memset(num_paths,0,n*sizeof(T));
   const INTM* pB = G.pB();
   const INTM* r = G.r();
   int* color = new int[n];
   memset(color,0,n*sizeof(int));
   list_int list;
   for (INTM i = 0; i<n; ++i)
      list.push_back(i);

   while (!list.empty()) {
      const INTM node=list.front();
      if (color[node]==0) {
         for (INTM i = pB[node]; i<pB[node+1]; ++i) {
            list.push_front(r[i]);
         }
         color[node]++;
      } else if (color[node]==1) {
         num_paths[node]=T(1.0);
         for (INTM i = pB[node]; i<pB[node+1]; ++i) {
            num_paths[node]+=num_paths[r[i]];
         }
         color[node]++;
      } else if (color[node]==2) {
         list.pop_front();
      }
   }

   T sum=cblas_asum<T>(n,num_paths,1);
   delete[](num_paths);
   delete[](color);
   return sum;
}


#endif

