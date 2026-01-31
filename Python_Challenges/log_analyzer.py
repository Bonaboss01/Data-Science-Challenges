def analyze_log(file_path):
    errors = 0
    warnings = 0

    with open(file_path, "r") as file:
        for line in file:
            if "ERROR" in line:
                errors += 1
            elif "WARNING" in line:
                warnings += 1

    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")


if __name__ == "__main__":
    analyze_log("app.log")
