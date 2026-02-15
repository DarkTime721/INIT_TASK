import torch
import torch.nn as nn

class SimpleANN(nn.Module):
    def __init__(self, input_features, hidden1, hidden2, output_features): #  classif = 'bin' (had this before)
        super(SimpleANN, self).__init__()

        self.model = nn.Sequential(
            nn.Flatten(),

            nn.Linear(input_features, hidden1),
            nn.ReLU(),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(),

            nn.Linear(hidden2, output_features)
        )
        '''
        if classif not in ("bin", "multi", "logits"):
            raise ValueError(
                "classif must be either 'bin' (binary) or 'multi' (multiclass)"
            )
        
        self.classif = classif
        '''
    def forward(self, x):

        x = self.model(x)

        '''
        if self.classif == 'bin':
            # For Binary Classification
            x = torch.sigmoid(x)
        elif self.classif == 'multi':
            # For Multi-Class classification
            x = F.softmax(x, dim=1)
        '''

        return x