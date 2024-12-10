import tensorflow as tf
try_button = tk.Button(root, text="Try", command=try_action)
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
import numpy as np

class RNNModel:
    def __init__(self, vocab_size, embedding_dim=256, rnn_units=1024):
        self.model = Sequential([
            # Embedding layer to convert tokens to vectors
            tf.keras.layers.Embedding(vocab_size, embedding_dim),
            
            # LSTM layer
            LSTM(rnn_units, return_sequences=True),
            LSTM(rnn_units//2),
            
            # Output layer
            Dense(vocab_size, activation='softmax')
        ])
        
        # Compile the model
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, x_train, y_train, epochs=10, batch_size=64):
        history = self.model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2
        )
        return history
    
    def generate_text(self, start_string, num_generate=1000, temperature=1.0):
        input_eval = [char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)
        text_generated = []

        self.model.reset_states()
        for i in range(num_generate):
            predictions = self.model(input_eval)
            predictions = tf.squeeze(predictions, 0)
            
            # Using temperature for prediction diversity
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
            
            input_eval = tf.expand_dims([predicted_id], 0)
            text_generated.append(idx2char[predicted_id])

        return start_string + ''.join(text_generated)

# Example usage:
# vocab_size = len(unique_chars)
# model = RNNModel(vocab_size)
# history = model.train(x_train, y_train)
# generated_text = model.generate_text("Starting text")

import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, n_layers=2):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM
        if hidden is None:
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(x.device)
            hidden = (h0, c0)
            
        output, hidden = self.lstm(embedded, hidden)
        
        # Reshape output for linear layer
        output = output.contiguous().view(-1, self.hidden_dim)
        output = self.fc(output)
        
        return output, hidden
    
    def generate_text(self, start_sequence, max_length=1000, temperature=1.0, device='cpu'):
        self.eval()
        with torch.no_grad():
            current = start_sequence
            hidden = None
            output_text = start_sequence
            
            for _ in range(max_length):
                # Prepare input
                x = torch.tensor([[current[-1]]]).to(device)
                
                # Get prediction
                output, hidden = self(x, hidden)
                
                # Apply temperature
                output = output / temperature
                probs = torch.softmax(output, dim=-1)
                
                # Sample from distribution
                next_char = torch.multinomial(probs, 1).item()
                
                output_text += idx2char[next_char]
                current = current[1:] + [next_char]
        
        return output_text

    def train_model(self, train_loader, criterion, optimizer, n_epochs, device='cpu'):
        self.train()
        for epoch in range(n_epochs):
            total_loss = 0
            hidden = None
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                output, hidden = self(inputs, hidden)
                hidden = tuple(h.detach() for h in hidden)
                
                # Calculate loss
                loss = criterion(output, targets.view(-1))
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

# Example usage:
# model = RNNModel(vocab_size=len(vocab))
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters())
# model.train_model(train_loader, criterion, optimizer, n_epochs=10)
# generated_text = model.generate_text(start_sequence=[...])
