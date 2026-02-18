import pytest

from alchemiscale import Scope
from alchemiscale.compute.settings import ComputeServiceSettings


class TestComputeServiceSettings:

    def test_validate_scopes(self):
        orgs = ["testorg", "*"]
        campaigns = ["testcompaign", "*"]
        projects = ["testproject", "*"]

        scopes = []
        for org in orgs:
            for campaign in campaigns:
                if org == "*" and campaign != "*":
                    continue
                for project in projects:
                    if campaign == "*" and project != "*":
                        continue
                    string_rep = f"{org}-{campaign}-{project}"
                    scopes.append(string_rep)
                    scopes.append(Scope.from_str(string_rep))

        assert ComputeServiceSettings.validate_scopes(scopes) == list(
            map(lambda s: Scope.from_str(s) if type(s) is str else s, scopes)
        )

        with pytest.raises(ValueError):
            scopes = ["*.*.*", "*-*-*"]
            ComputeServiceSettings.validate_scopes(scopes)

    def test_validate_scopes_exclude(self):
        orgs = ["testorg", "*"]
        campaigns = ["testcompaign", "*"]
        projects = ["testproject", "*"]

        scopes = []
        for org in orgs:
            for campaign in campaigns:
                if org == "*" and campaign != "*":
                    continue
                for project in projects:
                    if campaign == "*" and project != "*":
                        continue
                    string_rep = f"{org}-{campaign}-{project}"
                    scopes.append(string_rep)
                    scopes.append(Scope.from_str(string_rep))

        assert ComputeServiceSettings.validate_scopes_exclude(scopes) == list(
            map(lambda s: Scope.from_str(s) if type(s) is str else s, scopes)
        )

        with pytest.raises(ValueError):
            scopes = ["*.*.*", "*-*-*"]
            ComputeServiceSettings.validate_scopes_exclude(scopes)

    def test_validate_scopes_exclude_none(self):
        assert ComputeServiceSettings.validate_scopes_exclude(None) is None
