
This projects is based on code from the github repo
https://github.com/Infatoshi/mnist-cuda/tree/master
specifically the file v5.cu in this directory is the file this project is based off of.


Their are two main purposes for this project. 

1. To optimize the accuracy and performance of this code further and accomplish above 95% testing accuracy and significant performance advancements
This will be accomplished in a few ways.

    1. The first is a complete restructure of the network by adding two more hidden layers to the MLP to allow for more complex learning, that hopefully translates to better accuracy.
    2. The second optimization is a decaying learning rate that will allow the model to slow its learning as it gets more accurate in hopes of more accurate fine tuning
    3. A dropout rate of 20% was added to the relu kernel to help prevent overfitting and too much reliance on certain neurons. This should help with accuracy when we transfer over to a non-mnist data set because it helps with generalization
    4. add a weight decay constant to WEIGHT_DECAY to discourage overfitting and encourage generalization. 
    
Another small thing we need to do is setup a model weight saving and loading functionality because the current model just ends after every train and run session.

2. the second is two create a system for recognizing real handwritten digits in realtime from openCV footage

    The major problem that needs to be solved is creating a pipeline that can recognize numbers in webcam footage, capture them, then convert them to a 28x28 image that can be accurately predicted by the MNIST MLP.
    
    This has been accomplished with:
    - MSER-based digit detection in `camera/detectDigit.py`
    - Real-time camera processing pipeline
    - Model retraining system for camera-specific data in `camera/retraining/`
    
## Retraining System

The project includes a comprehensive retraining system that allows you to collect data from your camera and fine-tune the model for better accuracy on your specific setup:

- **Data Collection**: Capture digit samples from your camera stream
- **Interactive Labeling**: Manually label collected samples
- **Data Validation**: Quality checks and validation
- **Model Retraining**: Fine-tune the model with collected data

See `camera/retraining/RETRAINING_GUIDE.md` for detailed instructions.

### Quick Start for Retraining

```bash
# Run complete retraining workflow
make retrain-workflow

# Or run individual steps
make collect-data      # Collect samples from camera
make validate-data     # Validate data quality
make label-data        # Interactive labeling
make organize-data     # Split into train/val/test
make retrain-model     # Retrain the model
```