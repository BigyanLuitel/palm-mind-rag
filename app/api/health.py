from fastapi import APIRouter, Response

from app.api.deps import HealthCheckerDep
from app.services.health import HealthReport

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthReport,
    responses={503: {"model": HealthReport}},
)
async def health(checker: HealthCheckerDep, response: Response) -> HealthReport:
    report = await checker.check()
    if report.status != "ok":
        response.status_code = 503
    return report
