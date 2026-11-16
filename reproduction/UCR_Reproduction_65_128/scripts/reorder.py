heavy = ["NonInvasiveFetalECGThorax1", "NonInvasiveFetalECGThorax2", "StarLightCurves",
         "UWaveGestureLibraryAll", "SemgHandMovementCh2", "SemgHandSubjectCh2",
         "PhalangesOutlinesCorrect", "MixedShapesRegularTrain", "MelbournePedestrian"]
names = open("datasets_65_128.txt").read().split()
assert all(h in names for h in heavy)
ordered = [n for n in names if n not in heavy] + heavy
assert sorted(ordered) == sorted(names)
open("datasets_65_128.txt", "w").write("\n".join(ordered))
print(len(ordered), ordered[:3], ordered[-3:])