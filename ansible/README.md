# Ansible для agent-dialogue.ru

Три плейбука, каждый — отдельный шаг жизненного цикла:

| Когда          | Команда                                                     | Что делает                                 |
|----------------|-------------------------------------------------------------|--------------------------------------------|
| Один раз       | `ansible-playbook playbooks/bootstrap.yml`                  | Docker, юзер, ufw, репо, `.env`            |
| Один раз       | `ansible-playbook playbooks/tls.yml`                        | Выпуск Let's Encrypt cert (dummy-trick)    |
| Каждую выкатку | `ansible-playbook playbooks/deploy.yml -e image_tag=<sha>`  | Pull образа, restart, healthcheck          |

CI запускает только `deploy.yml`. `bootstrap.yml` и `tls.yml` выполняются вручную с локалки один раз.

## Установка локально

```bash
pip install "ansible-core>=2.16"
cd ansible
ansible-galaxy collection install -r requirements.yml
cp inventory.example.ini inventory.ini   # при необходимости поправьте host/user
```

Проверка коннекта:

```bash
ansible prod -m ping
```

## Полный сценарий с нуля

1. **DNS:** `agent-dialogue.ru` и `www.agent-dialogue.ru` → `176.108.249.8`.

2. **Bootstrap:**
   ```bash
   ansible-playbook playbooks/bootstrap.yml
   ```
   После этого ssh-нитесь на сервер и заполните `.env`:
   ```bash
   ssh agent@176.108.249.8 'sudo nano /opt/agent-dialogue/.env'
   ```
   Минимум: `OPENAI_API_KEY`.

3. **TLS-cert:**
   ```bash
   # Сначала проверим на staging-сервере Let's Encrypt (нет rate-limit):
   ansible-playbook playbooks/tls.yml -e le_staging=true

   # Когда убедились, что всё работает — боевой cert:
   ansible-playbook playbooks/tls.yml -e force_renew=true
   ```

4. **Первый деплой:**
   ```bash
   ansible-playbook playbooks/deploy.yml
   ```
   После этого `https://agent-dialogue.ru/api/health` должен отдавать `{"ok": true}`.

## Регулярные операции

### Дернуть конкретный SHA (откат)
```bash
ansible-playbook playbooks/deploy.yml -e image_tag=abc123def456
```

### Перевыпустить cert вручную
```bash
ansible-playbook playbooks/tls.yml -e force_renew=true
```
(certbot-контейнер в compose сам делает `renew` раз в 12 часов — этот шаг нужен только при смене домена / профилактике.)

### Только проверить состояние, ничего не менять
```bash
ansible-playbook playbooks/deploy.yml --check --diff
```

### Запустить что-то одно на сервере
```bash
ansible prod -m command -a 'docker compose ps' --become --become-user agent
ansible prod -m command -a 'docker compose logs --tail=50 app' --become --become-user agent
```

## CI-сторона

`.github/workflows/deploy.yml` использует environment `prod` со следующими secrets:

| Secret             | Назначение                                                 |
|--------------------|-----------------------------------------------------------|
| `SSH_PRIVATE_KEY`  | Приватный ключ, чьим public-полем настроен `agent@vps`     |
| `SSH_HOST`         | IP или DNS-имя сервера                                    |
| `SSH_USER`         | `agent`                                                   |
| `GITHUB_TOKEN`     | Авто-токен GH (push в GHCR — public package)              |

На каждый push в `main`:
1. CI собирает образ → пушит в `ghcr.io/ilya0217/agent-dialogue-sim:<sha>` + `:latest`.
2. CI рендерит inventory.ini из secrets, ставит ansible-core + community.general.
3. Запускает `playbooks/deploy.yml` с `image_tag=<sha>`.
4. Playbook сам делает smoke-тест `https://agent-dialogue.ru/api/health` через `uri:` (с локалки runner-а, через настоящий nginx).

## Структура

```
ansible/
  ansible.cfg
  requirements.yml
  inventory.example.ini       # шаблон
  inventory.ini               # локальный, в .gitignore
  group_vars/all.yml          # репо, домены, образ, le-настройки
  playbooks/
    bootstrap.yml
    tls.yml
    deploy.yml
  README.md                   # этот файл
```
