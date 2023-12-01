import torch                                                                                                                                                  
import torch.nn as nn                                                                                                                                         
                                                                                                                                                              
from torch.nn.utils.rnn import pack_padded_sequence                                                                                  
                                                                                                                                                              
torch.manual_seed(1)                                                                                                                                          
                                                                                                                                                              
lstm = nn.LSTM(3, 3).cuda()                                                                                                                                   
inputs = torch.randn((3,3,3)).to(device='cuda')                                                                                                               
                                                                                                                                                              
                                                                                                                                                              
packed = pack_padded_sequence(inputs, [3,3,3])                                                                                                                
out, hidden = lstm(packed)                                                                                                                                    
                                                                                                                                                              
print(out)                                                                                                                                                    
print(hidden)