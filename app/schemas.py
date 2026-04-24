from pydantic import BaseModel, Field


class HouseFeatures(BaseModel):
    GrLivArea:    float = Field(..., description="Above-ground living area (sq ft)", ge=300, le=6000)
    BedroomAbvGr: float = Field(..., description="Number of bedrooms", ge=0, le=10)
    FullBath:     float = Field(..., description="Number of full bathrooms", ge=0, le=5)
    OverallQual:  float = Field(..., description="Overall condition (1=very poor, 10=excellent)", ge=1, le=10)
    YearBuilt:    float = Field(..., description="Year the house was built", ge=1872, le=2010)

    def to_array(self) -> list[float]:
        return [self.GrLivArea, self.BedroomAbvGr, self.FullBath, self.OverallQual, self.YearBuilt]


class PredictRequest(BaseModel):
    features: HouseFeatures


class PredictResponse(BaseModel):
    prediction: float
    prediction_usd: str


class UncertaintyResponse(BaseModel):
    prediction: float
    prediction_usd: str
    lower_bound: float
    upper_bound: float
    lower_bound_usd: str
    upper_bound_usd: str
    interval_width: float
    margin: float
    confidence_level: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    calibrated: bool
    task_type: str
