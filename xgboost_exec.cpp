
		void find_greedy_split(int node_idx){
			
			std::vector<double> gradientS;
			std::vector<double>  hessianS;
			
			std::vector<int> lhsF;
			std::vector<int>  rhsF;
			
			for(int i : tree[node_idx].idxs){
				gradientS.emplace_back(this->gradient[i]);
				hessianS.emplace_back(this->hessian[i]);
				//std::cout<<i<<" ";		
			}
			//std::cout<<std::endl;
			
			
			int node_tree_size = tree[node_idx].idxs.size();
			
			if( node_tree_size <= 1 || tree[node_idx].depth >= this->depth-1){
				tree[node_idx].val = this->compute_gamma(gradientS,hessianS);
				return;
			}
			
			
			for (int c : this->column_subsample){
				std::vector<double> xsplit;
				
				// on selectionne les valeurs d'une caracteristiques donnee suivant un certain nombre d'indice
				for (int idx : tree[node_idx].idxs){
					xsplit.emplace_back(this->x[idx][c]);
				}
				int xsplit_size = xsplit.size();
				
				for (int r = 0; r < xsplit_size; r++){
					std::vector<bool> lhs(xsplit_size, false);
					std::vector<bool> rhs(xsplit_size, false);
					int lhs_sum=0;
					int rhs_sum=0;
					std::vector<int> lhs_indices, rhs_indices;
					double lhs_hessian_sum = 0.0, rhs_hessian_sum = 0.0;
					
					for (int i = 0; i < xsplit_size; i++){
						if (xsplit[i] <= xsplit[r]) {
							lhs[i] = true;
							lhs_sum++;
							lhs_indices.emplace_back(tree[node_idx].idxs[i]);
							lhs_hessian_sum += hessianS[i];
						} else{
							rhs[i] = true;
							rhs_sum++;
							rhs_indices.emplace_back(tree[node_idx].idxs[i]);
							rhs_hessian_sum += hessianS[i];
						}
						
					}
					
					
					
					if (rhs_sum < this->min_leaf || lhs_sum < this->min_leaf || rhs_hessian_sum < this->min_child_weight || lhs_hessian_sum < this->min_child_weight ) {
						continue;
					}
					
					
					double  curr_score = this->gain(lhs,rhs,node_idx);
					
					if (curr_score > tree[node_idx].score){
						tree[node_idx].var_idx = c;
						tree[node_idx].score  = curr_score;
						tree[node_idx].split = xsplit[r];
						lhsF.clear();
						rhsF.clear();
						lhsF = lhs_indices;
						rhsF = rhs_indices;
					}
					lhs_indices.clear();
					rhs_indices.clear();
					lhs.clear();
					rhs.clear();
				}
			
				xsplit.clear();
			}
			//ajout des noeuds gauche et droit dans notre arbre
			int left_child_idx = tree.size();
			tree.emplace_back(Node(lhsF,tree[node_idx].depth+1,0.0, -10000.0,-1, 0.0, -1,-1));
			int right_child_idx = tree.size();
			tree.emplace_back(Node(rhsF,tree[node_idx].depth+1,0.0, -10000.0,-1, 0.0, -1,-1));
			
			//std::cout<<node_idx<<" "<<node_tree_size<<" "<<tree[node_idx].idxs[0]<<" "<<tree[node_idx].idxs[1]<<" "<<tree.size()<<std::endl;
			
			tree[node_idx].left = left_child_idx;
			tree[node_idx].right = right_child_idx;
		
			node_stack.push(left_child_idx);
			node_stack.push(right_child_idx);
			
		}
