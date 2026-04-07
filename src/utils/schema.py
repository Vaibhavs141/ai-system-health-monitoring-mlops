from pydantic import BaseModel, Field


class RawSystemHealthRecord(BaseModel):
    CPUUsage: float = Field(..., ge=0, le=100)
    RAMUsage: float = Field(..., ge=0, le=100)
    Temperature: float = Field(..., ge=0, le=150)
    Voltage: float = Field(..., ge=0, le=20)
    DiskUsage: float = Field(..., ge=0, le=100)
    FanSpeed: float = Field(..., ge=0)
    ProblemDetected: str