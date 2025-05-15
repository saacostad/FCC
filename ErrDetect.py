import pandas as pd

MAGNET_PARAMETERS = ["K1", "L"]

file_path = "data/fccee_t.seq"

inLines = dict.fromkeys(["NAME"] + MAGNET_PARAMETERS)
df = None


nameFlag = True
for param in MAGNET_PARAMETERS:
    nameFlag = True

    with open(file_path, "r") as file:
        lines = [line for line in file if line.startswith(f"{param + 'Q'}")]

        names = list()
        values = list()

        for line in lines:
            name, value = line[len(param) :].split(" = ")
            value = value.replace(";\n", "")
            names.append(name)
            values.append(float(value))

        if nameFlag:
            inLines["NAME"] = names
            nameFlag = False

        inLines[param] = values


df = pd.DataFrame(inLines)

print(df)
