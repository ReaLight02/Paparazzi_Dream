import data_loader

TrainClasses = data_loader.training_set.class_indices

# Mapping face : id in ResultMap
ResultMap = {}
for faceValue, faceName in zip(TrainClasses.values(), TrainClasses.keys()):
    ResultMap[faceValue] = faceName

print("Mapping Face : Id", ResultMap)

OutputNeurons = len(ResultMap)
print("\n Output neurons: ", OutputNeurons)
