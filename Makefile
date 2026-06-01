.PHONY: e2e e2e-update e2e-clean

e2e:
	./scripts/e2e.sh check

e2e-update:
	./scripts/e2e.sh update

e2e-clean:
	rm -rf target/e2e
