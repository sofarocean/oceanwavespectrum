def run_input_tests(function_to_test, testdata):
    for kw, test in testdata.items():
        try:
            _ = function_to_test(**test["kwargs"])
            # Assert that this should not raise an error
            if test["errorstring"] is not None:
                raise ValueError(f"Test {kw} should have raised an error.")
        except Exception as e:
            if test["errorstring"] is None:
                # Unexpected error
                raise e
            else:
                if test["errorstring"] != e.args[0]:
                    # Genuine error
                    raise e
