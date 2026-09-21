from scripts.PrepareSkillsbench import RewritePackageManagerRuns


def test_druid_archive_download_is_parallel_and_ordered():
    dockerfile = """
FROM ubuntu:24.04
RUN cd /tmp && wget https://archive.apache.org/dist/druid/0.20.0/apache-druid-0.20.0-bin.tar.gz && tar -xzf apache-druid-0.20.0-bin.tar.gz
"""

    rendered = RewritePackageManagerRuns(dockerfile)

    assert "xargs -P 8" in rendered
    assert "part-%08d" in rendered
    assert 'test "$(wc -c < "$archive_output")" -eq "$archive_size"' in rendered


def test_github_download_with_version_variable_has_proxy_fallback_and_resume():
    dockerfile = """
FROM ubuntu:24.04
RUN OTP_DOWNLOAD_URL="https://github.com/erlang/otp/releases/download/OTP-${OTP_VERSION}/otp_src_${OTP_VERSION}.tar.gz" \\
    && curl -fSL -o otp-src.tar.gz "$OTP_DOWNLOAD_URL" \\
    && sha256sum otp-src.tar.gz
"""

    rendered = RewritePackageManagerRuns(dockerfile)

    assert "gh-proxy.com/https://github.com/erlang/otp" in rendered
    assert "OTP_DOWNLOAD_URL_FALLBACK" in rendered
    assert "s/gh-proxy\\.com/gh-proxy\\.org/" in rendered
    assert "--continue-at -" in rendered
