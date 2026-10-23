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


candidate.parquet
item_id  retrieval_id                                      100  \
0  20000006363             0  {'cold_start': 0, 'feature_value': '6'}   
1  20000016200             1  {'cold_start': 0, 'feature_value': '6'}   
2  20000016475             2  {'cold_start': 0, 'feature_value': '4'}   
3  20000018082             3  {'cold_start': 0, 'feature_value': '6'}   
4  20000018288             4  {'cold_start': 0, 'feature_value': '6'}   

                                        101  \
0  {'cold_start': 0, 'feature_value': '42'}   
1  {'cold_start': 0, 'feature_value': '22'}   
2   {'cold_start': 0, 'feature_value': '9'}   
3  {'cold_start': 0, 'feature_value': '17'}   
4  {'cold_start': 0, 'feature_value': '42'}   

                                           102  \
0  {'cold_start': 0, 'feature_value': '63599'}   
1  {'cold_start': 0, 'feature_value': '46781'}   
2  {'cold_start': 0, 'feature_value': '61049'}   
3  {'cold_start': 0, 'feature_value': '52815'}   
4  {'cold_start': 0, 'feature_value': '21061'}   

                                        112  \
0  {'cold_start': 0, 'feature_value': '29'}   
1  {'cold_start': 0, 'feature_value': '29'}   
2  {'cold_start': 0, 'feature_value': '28'}   
3  {'cold_start': 0, 'feature_value': '29'}   
4  {'cold_start': 0, 'feature_value': '29'}   

                                       114  \
0  {'cold_start': 0, 'feature_value': '3'}   
1  {'cold_start': 0, 'feature_value': '3'}   
2  {'cold_start': 0, 'feature_value': '8'}   
3  {'cold_start': 0, 'feature_value': '3'}   
4  {'cold_start': 0, 'feature_value': '3'}   

                                           115  \
0  {'cold_start': None, 'feature_value': None}   
1    {'cold_start': 0, 'feature_value': '368'}   
2  {'cold_start': None, 'feature_value': None}   
3    {'cold_start': 0, 'feature_value': '431'}   
4  {'cold_start': None, 'feature_value': None}   

                                        116  \
0   {'cold_start': 0, 'feature_value': '7'}   
1   {'cold_start': 0, 'feature_value': '2'}   
2  {'cold_start': 0, 'feature_value': '12'}   
3   {'cold_start': 0, 'feature_value': '2'}   
4   {'cold_start': 0, 'feature_value': '7'}   

                                         117  \
0  {'cold_start': 0, 'feature_value': '389'}   
1  {'cold_start': 0, 'feature_value': '201'}   
2  {'cold_start': 0, 'feature_value': '425'}   
3  {'cold_start': 0, 'feature_value': '167'}   
4   {'cold_start': 0, 'feature_value': '33'}   

                                          118  \
0   {'cold_start': 0, 'feature_value': '572'}   
1   {'cold_start': 0, 'feature_value': '107'}   
2  {'cold_start': 0, 'feature_value': '1425'}   
3   {'cold_start': 0, 'feature_value': '480'}   
4   {'cold_start': 0, 'feature_value': '519'}   

                                          119  \
0   {'cold_start': 0, 'feature_value': '667'}   
1  {'cold_start': 0, 'feature_value': '3116'}   
2  {'cold_start': 0, 'feature_value': '1578'}   
3  {'cold_start': 0, 'feature_value': '1762'}   
4  {'cold_start': 0, 'feature_value': '3766'}   

                                          120  \
0   {'cold_start': 0, 'feature_value': '621'}   
1  {'cold_start': 0, 'feature_value': '1124'}   
2  {'cold_start': 0, 'feature_value': '3313'}   
3   {'cold_start': 0, 'feature_value': '640'}   
4  {'cold_start': 0, 'feature_value': '3249'}   

                                             121  \
0   {'cold_start': 0, 'feature_value': '794587'}   
1  {'cold_start': 0, 'feature_value': '1821550'}   
2  {'cold_start': 0, 'feature_value': '1980788'}   
3   {'cold_start': 0, 'feature_value': '857512'}   
4   {'cold_start': 0, 'feature_value': '365152'}   

                                           122  
0  {'cold_start': 0, 'feature_value': '28134'}  
1  {'cold_start': 0, 'feature_value': '80452'}  
2  {'cold_start': 0, 'feature_value': '13072'}  
3  {'cold_start': 0, 'feature_value': '13825'}  
4  {'cold_start': 0, 'feature_value': '62653'}


i.json
{
  "20004268480": 1,
  "20002049346": 2,
  "20001902532": 3,
  "20004618112": 4,
  "20002361362": 5,
  "20002381090": 6,
  "20002018670": 7,
  "20005581540": 8,
  "20003828718": 9,
  "20002915204": 10,
  ...