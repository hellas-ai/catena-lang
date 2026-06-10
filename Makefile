.PHONY: e2e e2e-update e2e-clean lang-test lang-test-update lang-test-clean

e2e:
	./scripts/e2e.sh check

e2e-update:
	./scripts/e2e.sh update

e2e-clean:
	rm -rf target/e2e

lang-test:
	./tests/lang/run.sh check

lang-test-update:
	./tests/lang/run.sh update

lang-test-clean:
	rm -rf target/catena-lang-tests/lang
