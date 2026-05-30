from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import Literal


class ExternalCreditSource(BaseModel):

    EXT_SOURCE_1: float = Field(ge=0, le=1, description='External scores from source 1')
    EXT_SOURCE_3: float = Field(ge=0, le=1, description='External scores from source 3')
    EXT_SOURCE_WEIGHTED: float = Field(ge=0, le=1, description='Weighted external credit score')


class LoanInputs(BaseModel):

    AMT_CREDIT: float = Field(description='amount credit of loan')
    AMT_GOODS_PRICE: float = Field(description='goods price for loan is applied')
    GOODS_CREDIT_RATIO: float = Field(description='Ratio of goods price to credit amount')

    @model_validator(mode='after')
    def validate_credit_lt_goods(self):
        if self.AMT_CREDIT < self.AMT_GOODS_PRICE:
            raise ValueError('Loan amount cannot exceed goods price')
        return self


class PersonalInfo(BaseModel):

    YEARS_AGE: int = Field(ge=18, lt=65, description='Enter Your age')
    NAME_FAMILY_STATUS: Literal["Married", "Single / not married",
                                "Civil marriage", "Separated", "Widow", "Unknown"] = Field(description="Marital status of applicant")
    FLAG_DOCUMENT_3: int = Field(description='Flag document 3')
    REGION_RATING_CLIENT_W_CITY: Literal[1, 2, 3] = Field(description='Region rating where client lives')


class EmploymentPayementFeatures(BaseModel):

    YEARS_EMPLOYED: int = Field(ge=0, le=60, description='total years borrower is employed')
    NAME_INCOME_TYPE: Literal["Working", "Commercial associate", "State servant",
                              "Pensioner", "RARE"] = Field(description='borrowers income type')
    IP_WORST_DPD_720D: int = Field(ge=0, le=690, description='enter worst dpd in last 720 days')


class BureauFeatures(BaseModel):

    B_NUM_ACTIVE_CREDIT_720D: int = Field(ge=0, lt=10, description='number active loans user have in last 720 Days')
    B_DEBT_TO_CREDIT_RATIO: float = Field(description='Total amount approved all the active loans')
    B_CREDIT_DURATION_MIN: float = Field(description='What is shortest duration loan the customer had')

    @model_validator(mode='after')
    def validate_credit_lt_debt(self):
        if self.B_DEBT_TO_CREDIT_RATIO > 1:
            raise ValueError('debt ratio cannot exceed 1')
        return self


class InstallmentFeatures(BaseModel):

    IP_RATIO_LATE_PAYMENTS_2160D: float = Field(description='Ratio of late payments over 2160 days higher values mean more late payments and higher credit risk.')
    IP_NUM_COMPLETED_LOANS: int = Field(ge=0, le=20, description='Number of completed loans')


class CreditAtmFeatures(BaseModel):

    CB_AVG_ATM_WITHDRAWAL_FREQ_6M: float = Field(le=35, description='Avg ATM withdrawal count per month over last 6 months · data max ≈ 35')
    CB_WT_CREDIT_UTIL_TREND_3M_12M: float = Field(description='credit utilisation trend Change in how much credit the customer is using recently')


class PreviousApplicationFeatures(BaseModel):

    PA_RATIO_APPROVED_LOANS: float = Field(gt=0, le=1, description='Ratio of past approved loan applications')
    PA_AVG_AMT_ANNUITY_POS: float = Field(description='Average annuity(EMI) amount on POS previous applications')
    PA_AVG_RISK_WEIGHT_1080D: float = Field(description='Average risk level of past loan applications in the last ~3 years; higher values indicate riskier past borrowing.')
    PA_RATIO_CREDIT_APPLICATION_Cash: float = Field(description='for CASH LOANS Credit given vs applied amount (bank approval behavior).')
    PA_RATIO_CREDIT_APPLICATION_POS: float = Field(description='for POS LOANS Credit given vs applied amount (bank approval behavior).')


class CategoricalFeatures(BaseModel):

    OCCUPATION_GROUP: Literal["LOW_SKILL", "SKILLED_PRO", "MANAGERS", "SERVICE", "MISSING"] = Field(description='in which occupation borrower belongs')
    ORG_GROUP: Literal["ORG_PRIVATE", "ORG_STABLE", "ORG_OTHER", "ORG_UNSTABLE"] = Field(description='in which org group borrower belongs')
    NAME_EDUCATION_TYPE: Literal["Higher education", "Secondary / secondary special",
                                 "Incomplete higher", "Lower secondary",
                                 "Academic degree", "MISSING"] = Field(description='name borrowers highest education')
    CODE_GENDER: Literal['M', 'F'] = Field(description='name borrowers gender')


class CreditApplicationRequest(BaseModel):

    model_config = ConfigDict(str_strip_whitespace=True)

    external_credit_sources: ExternalCreditSource
    loan_inputs: LoanInputs
    personal_info: PersonalInfo
    employment_payment_history_feature: EmploymentPayementFeatures
    bureau_features: BureauFeatures
    installment_features: InstallmentFeatures
    credit_atm_features: CreditAtmFeatures
    previous_application_features: PreviousApplicationFeatures
    categorical_features: CategoricalFeatures