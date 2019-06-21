import torch
import torch.nn as nn
from lstm import LSTM


class LSTM_sentiment(nn.Module):
    def __init__(self, vocab_len, embed_dim, hidden_dim, output_dim, pretrained_vec,
                 layers=1, bidirectional=False, xavier_init=False):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_len, embed_dim)
        # not training embedding layer if pretrained embedding is provided
        if pretrained_vec is not None:
            self.embedding.weight.data.copy_(pretrained_vec)
            self.embedding.weight.requires_grad = False

        self.bidirectional = bidirectional
        self.lstm = LSTM(input_dim=embed_dim, hidden_dim=hidden_dim, layers=layers,
                         bidirectional=bidirectional, xavier_init=xavier_init)
        if self.bidirectional:
            self.fc = nn.Linear(2 * hidden_dim, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden_state, cell_state):
        embed = self.embedding(x)
        _, (hidden_state, _) = self.lstm(embed, hidden_state, cell_state)
        if self.bidirectional:
            # Flattening hidden state for the 2 directions in bidirectional
            hidden_state = torch.cat((hidden_state[:,0,:,:], hidden_state[:,1,:,:]), dim=2)
        hidden_state = hidden_state[-1]
        output = self.fc(hidden_state)
        return output

    def count_parameters(self):
        tot_sum = sum(p.numel() for p in self.lstm.parameters() if p.requires_grad)
        tot_sum += sum(p.numel() for p in self.fc.parameters() if p.requires_grad)
        return tot_sum


def train_sentiment(model, train_loader, device, epochs, optimizer, loss_criterion, valid_loader=None):
    """
    Trains the model of class LSTM using data from data_loader passed as argument

    Parameters
    ==========
    model: object of class torch.nn.Module
    train_loader: an object of class BucketIterator containing training data
        The next() function of the object returns a batch object, which has 2 members
        text   - which is a 3D tensor of [sequence, batch, dim]
            Example-A batch of 10 sentences of 5 words each where each word has
            an embedding vector of size 256, the 3D tensor shape will be [5, 10, 256]
        labels - a 1D tensor of length batch_size containing the classes as integers (torch.long())
    valid_loader: Optional parameter, an object of class BucketIterator containing validation data.
        Similar to train_loader
        If passed, then it validates with the given data
    Returns
    =======
    Trained model object of class LSTM, list containing loss progress
    (, and list containing validation F1 score)

    """

    stats = {'loss': [], 'train_score': [], 'valid_score': []}
    for i in range(epochs):
        loss_tracker = []

        model.train()
        for j, batch in enumerate(train_loader):
            # generate initial hidden & cell states
            hidden_state = torch.zeros(1, batch.label.shape[0],
                                       model.hidden_dim, requires_grad=True).to(device)
            cell_state = torch.zeros(1, batch.label.shape[0],
                                     model.hidden_dim, requires_grad=True).to(device)

            # forward pass
            output = model(batch.text, hidden_state, cell_state)
            # backward pass for the batch (+ weight updates)
            optimizer.zero_grad()
            loss = loss_criterion(output.squeeze(1), batch.label)
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 1)
            optimizer.step()

            print("Epoch #{}: Batch {}/{} -- Loss = {}".format(i + 1, j + 1, len(train_loader),
                                                               loss.item()), end='\r')
            loss_tracker.append(loss.item())

        stats['loss'].append(np.mean(loss_tracker))
        loss_tracker = []
        print()
        print("Epoch #{}: Average loss is {}".format(i + 1, stats['loss'][-1]))
        if i % 2 == 0:
            f1 = evaluate_sentiment(model, train_loader, device, verbose=False)
            stats['train_score'].append(f1)
            print("Epoch #{}: Train F1-score is {}".format(i + 1, stats['train_score'][-1]))
        if i % 2 == 0 and valid_loader is not None:
            f1 = evaluate_sentiment(model, valid_loader, device, verbose=False)
            stats['valid_score'].append(f1)
            print("Epoch #{}: Validation F1-score is {}".format(i + 1, stats['valid_score'][-1]))
        print()
    return model, stats


def evaluate_sentiment(model, test_loader, device, verbose=True):
    """
    Evaluates the model of class LSTM using test data passed

    Parameters
    ==========
    model: object of class LSTM
    test_data: a tuple containing (x, y)
        x - a 3D tensor of [sequence, len_test_data, dim]
            Example-A batch of 10 sentences of 5 words each where each word has
            an embedding vector of size 256, the 3D tensor shape will be [5, 10, 256]
        y - a 1D tensor containing the classes as integers (torch.long())
    verbose: prints the confusion matrix and F-score

    Returns
    =======
    F-score (float)

    """
    model.eval()

    preds = []
    labels = []

    with torch.no_grad():
        for batch in test_loader:
            hidden_state = torch.zeros(1, batch.label.shape[0], model.hidden_dim).to(device)
            cell_state = torch.zeros(1, batch.label.shape[0], model.hidden_dim).to(device)
            output = model(batch.text, hidden_state, cell_state)
            # get label predictions - since we predict only probabilities for label 1
            pred = torch.round(torch.sigmoid(output)).cpu().detach().numpy()
            preds.extend(pred)
            labels.extend(batch.label.cpu().detach().numpy())

    if verbose:
        print('Confusion Matrix: \n', confusion_matrix(labels, preds))
        print('Classification Report: \n', classification_report(labels, preds))
    return f1_score(labels, preds)
