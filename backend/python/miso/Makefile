.PHONY: miso
miso:
	bash install.sh

.PHONY: run
run: miso
	@echo "Running miso..."
	bash run.sh
	@echo "miso run."

.PHONY: test
test: miso
	@echo "Testing miso..."
	bash test.sh
	@echo "miso tested."

.PHONY: protogen-clean
protogen-clean:
	$(RM) backend_pb2_grpc.py backend_pb2.py

.PHONY: clean
clean: protogen-clean
	rm -rf venv __pycache__
