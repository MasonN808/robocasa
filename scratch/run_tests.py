import unittest
suite=unittest.TestLoader().discover("tests",pattern="test_*.py")
res=unittest.TextTestRunner(verbosity=2).run(suite)
print(f"\nSUMMARY run={res.testsRun} failures={len(res.failures)} errors={len(res.errors)}")
for t,_ in res.failures+res.errors: print("  FAIL:",t)
