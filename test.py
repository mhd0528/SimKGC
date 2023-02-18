#import torch module
import torch
 
 
#create  tensor
data1 = torch.tensor([[2,3,4,5],[1,3,5,3],[2,3,2,1],[2,3,4,2]])  
 
#display
print("Actual Tensor: ")
print(data1)
 
print("Cumulative Product across column: ")
#return cumulative Product
# print(torch.cumprod(data1,0)[-1])

print(torch.cumprod(torch.index_select(data1, 0, torch.Tensor([1, 2]).to(torch.int32)), dim=0)[-1])

test1 = torch.tensor([1, 2, 3, 4])
test2 = torch.tensor([1, 2, 3, 4])

# print(torch.mm(test1.unsqueeze(0), test2.unsqueeze(1)).squeeze())
tmp = torch.tensor([0.5 for _ in range(10)])
tmp[0] += torch.mm(test1.unsqueeze(0), test2.unsqueeze(1)).squeeze()
print(tmp)
