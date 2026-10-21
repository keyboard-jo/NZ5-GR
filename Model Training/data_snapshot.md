index_rqvae_prod.json
{"26": [76, 73, 51, 14, 0], "3492133": [76, 73, 51, 14, 1], "521243": [76, 73, 51, 14, 2], ...

seq.parquet (sequence data for each user and seq column. each seq.column has item_id, action_type, and timestamp in each element)
user_id  \
0   953188   
1   919490   
2   771916   
3   175810   
4   830016   

                                                                               seq  
0  [{'item_id': 2905777, 'action_type': 0, 'timestamp': 1745919240}, {'item_id'...  
1  [{'item_id': 2634013, 'action_type': 0, 'timestamp': 1729078089}, {'item_id'...  
2  [{'item_id': 2469426, 'action_type': 0, 'timestamp': 1743841472}, {'item_id'...  
3  [{'item_id': 2296459, 'action_type': 0, 'timestamp': 1747312952}, {'item_id'...  
4  [{'item_id': 429216, 'action_type': 1, 'timestamp': 1744283532}, {'item_id':...

user_feat.parquet (user_id with their respective features)
user_id  103  104  105   106   107   108  109   110
0       19   36    2    7   [6]  [15]  <NA>    2  <NA>
1       26   22    1    3  <NA>  <NA>  <NA>    1  <NA>
2       29   82    2    7   [6]   [9]   [2]    2  <NA>
3       54   44    1    7  [13]  <NA>  <NA>    2  <NA>
4       65    4    2    3   [4]  <NA>  <NA>    3  <NA>