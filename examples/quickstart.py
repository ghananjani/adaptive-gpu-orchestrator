from adaptive_gpu_orchestrator import Doctor


doctor = Doctor()
snapshot = doctor.inspect()
print(snapshot.to_dict())
print(doctor.diagnose().to_dict())
