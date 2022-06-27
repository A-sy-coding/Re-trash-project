#!/bin/bash
echo "update pip..."
pip3 install --upgrade pip

echo "EfficientNet git clone..."
git clone https://github.com/lukemelas/EfficientNet-PyTorch

echo "install requirements.txt..."
pip3 install -r requirements.txt
